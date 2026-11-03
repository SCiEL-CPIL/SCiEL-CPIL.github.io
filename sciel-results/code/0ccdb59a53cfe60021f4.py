import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 135.0
M_FFT = 2048  # Next power of 2 >= 2*600-1 = 1199
X_GRID = np.linspace(0, 1, N)

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _parametric_heights(center, width, taper_power, amplitude):
    """Super-ellipse shape: amp * (1 - (|x-c|/w)^p)^(1/p) for |x-c|<w, else 0.
    p=1: triangular, p=2: parabolic, p->inf: rectangular (top-hat).
    """
    d = np.abs(X_GRID - center)
    ratio = d / max(width, 1e-10)
    mask = ratio < 1.0
    heights = np.zeros(N)
    p = max(taper_power, 0.1)
    heights[mask] = amplitude * (1.0 - ratio[mask] ** p) ** (1.0 / p)
    return np.maximum(heights, 0.0)

def _smooth_max(conv, alpha):
    """Log-sum-exp smooth max."""
    m = np.max(conv)
    if m < 0:
        m = 0.0
    exp_conv = np.exp(alpha * (conv - m))
    return m + np.log(np.sum(exp_conv)) / alpha

def _smooth_max_weights(conv, alpha):
    """Weights (softmax) for smooth max gradient."""
    m = np.max(conv)
    if m < 0:
        m = 0.0
    exp_conv = np.exp(alpha * (conv - m))
    return exp_conv / np.sum(exp_conv)

def _smooth_obj_and_grad(heights, alpha):
    """Smooth objective and gradient using FFT-based correlation."""
    h = np.maximum(heights, 0.0)
    n = len(h)
    total = np.sum(h)
    if total <= 1e-10:
        return 1e10, np.zeros(n)
    
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = h
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    sm = _smooth_max(conv, alpha)
    weights = _smooth_max_weights(conv, alpha)
    
    # Gradient of smooth_max w.r.t. h:
    # d(sm)/d(h_i) = sum_k weights[k] * d(conv[k])/d(h_i)
    # d(conv[k])/d(h_i) = h[k-i] + h[i-k] (linear, with bounds)
    # = 2 * correlation(weights, h) at offset i
    W_fft = np.fft.rfft(weights)
    corr = np.fft.irfft(W_fft * np.conj(F), n=M_FFT)
    grad_f = 2.0 * corr[:n]
    
    w = 0.5 / n
    obj = sm * n / (total * total * w)
    obj_grad = grad_f * (n / (total * total * w)) - sm * (2.0 * n / (total * total * total * w))
    
    return obj, obj_grad

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0
    
    rng = np.random.default_rng(42)
    
    # Start with uniform
    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_score(best_h)
    
    # Phase 1: Coarse parametric search over super-ellipse family
    center_vals = np.linspace(0.3, 0.7, 7)
    width_vals = np.linspace(0.2, 1.0, 7)
    taper_vals = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 16.0, 32.0]
    amp_vals = [0.5, 1.0, 1.5]
    
    best_param = None
    best_param_score = best_score
    
    for c in center_vals:
        if deadline_reached():
            break
        for wdt in width_vals:
            if deadline_reached():
                break
            for tp in taper_vals:
                if deadline_reached():
                    break
                for amp in amp_vals:
                    h = _parametric_heights(c, wdt, tp, amp)
                    if np.sum(h) <= 0:
                        continue
                    s = compute_score(h)
                    if s < best_score:
                        best_score = s
                        best_h = h.copy()
                    if s < best_param_score:
                        best_param_score = s
                        best_param = (c, wdt, tp, amp)
    
    # Phase 2: Nelder-Mead refinement on parametric space
    if time_left() > 25 and best_param is not None:
        def param_obj(params):
            c, wdt, tp, amp = params
            c = np.clip(c, 0.1, 0.9)
            wdt = np.clip(wdt, 0.05, 1.0)
            tp = max(tp, 0.5)
            amp = max(amp, 0.01)
            h = _parametric_heights(c, wdt, tp, amp)
            if np.sum(h) <= 0:
                return 1e10
            return compute_score(h)
        
        try:
            res = minimize(param_obj, list(best_param), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            c, wdt, tp, amp = res.x
            c = np.clip(c, 0.1, 0.9)
            wdt = np.clip(wdt, 0.05, 1.0)
            tp = max(tp, 0.5)
            amp = max(amp, 0.01)
            h = _parametric_heights(c, wdt, tp, amp)
            s = compute_score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
        except Exception:
            pass
    
    # Phase 3: Multi-start smooth-max L-BFGS-B refinement
    # Build diverse starting points
    starts = [best_h.copy()]
    
    # Add parametric variations around best
    if best_param is not None:
        c0, w0, tp0, amp0 = best_param
        for dc in [-0.05, 0.05]:
            for dw in [-0.05, 0.05]:
                for dtp in [-1.0, 1.0]:
                    c = np.clip(c0 + dc, 0.1, 0.9)
                    wdt = np.clip(w0 + dw, 0.05, 1.0)
                    tp = max(tp0 + dtp, 0.5)
                    amp = max(amp0, 0.01)
                    h = _parametric_heights(c, wdt, tp, amp)
                    if np.sum(h) > 0:
                        starts.append(h)
    
    # Add random sparse starts
    for _ in range(5):
        hr = np.zeros(N)
        mask = rng.random(N) < 0.4
        hr[mask] = 1.0
        if np.sum(hr) > 0:
            starts.append(hr)
    
    # Add cosine-tapered starts
    for tf in [0.1, 0.2, 0.3]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        starts.append(ht)
    
    # Deduplicate and sort by score
    scored_starts = []
    for hs in starts:
        hs = np.maximum(hs, 0.0)
        if np.sum(hs) <= 0:
            continue
        s = compute_score(hs)
        scored_starts.append((s, hs))
    scored_starts.sort(key=lambda x: x[0])
    
    # Run L-BFGS-B on top starts with increasing alpha
    alpha_schedule = [30.0, 80.0, 200.0, 500.0]
    
    for alpha in alpha_schedule:
        if deadline_reached():
            break
        
        num_starts = min(4, len(scored_starts))
        for _, hs in scored_starts[:num_starts]:
            if deadline_reached():
                break
            
            def obj_wrapper(h):
                o, _ = _smooth_obj_and_grad(h, alpha)
                return o
            
            def grad_wrapper(h):
                _, g = _smooth_obj_and_grad(h, alpha)
                return g
            
            try:
                res = minimize(obj_wrapper, hs.copy(), jac=grad_wrapper,
                             method='L-BFGS-B',
                             bounds=[(0, None)] * N,
                             options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8,
                                      'maxls': 20})
                new_h = np.maximum(res.x, 0.0)
                new_score = compute_score(new_h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = new_h.copy()
                    # Update scored_starts for next alpha
                    scored_starts = [(new_score, new_h.copy())] + \
                                     [x for x in scored_starts if x[1] is not new_h]
                    scored_starts.sort(key=lambda x: x[0])
            except Exception:
                pass
    
    # Phase 4: Exact gradient descent with true max (final polish)
    if time_left() > 10:
        current_h = best_h.copy()
        current_score = best_score
        
        for _ in range(200):
            if deadline_reached():
                break
            
            h_norm = current_h / np.sum(current_h)
            f = np.zeros(M_FFT, dtype=np.float64)
            f[:N] = h_norm
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=M_FFT)
            k_star = np.argmax(conv)
            
            # Gradient of max at k_star
            grad_conv = np.zeros(N)
            idx1 = k_star - np.arange(N)
            valid1 = (idx1 >= 0) & (idx1 < N)
            grad_conv[valid1] += h_norm[idx1[valid1]]
            idx2 = np.arange(N) - k_star
            valid2 = (idx2 >= 0) & (idx2 < N)
            grad_conv[valid2] += h_norm[idx2[valid2]]
            
            w = 0.5 / N
            grad_norm = (N / w) * grad_conv
            total = np.sum(current_h)
            sum_grad_h = np.dot(grad_norm, current_h)
            grad = grad_norm / total - current_h / (total * total) * sum_grad_h
            
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-10:
                break
            grad = grad / gnorm
            
            step_size = 0.05
            improved = False
            for _ in range(8):
                new_h = np.maximum(current_h - step_size * grad, 0.0)
                if np.sum(new_h) <= 0:
                    step_size *= 0.5
                    continue
                new_score = compute_score(new_h)
                if new_score < current_score:
                    current_h = new_h
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
                    improved = True
                    break
                step_size *= 0.5
            
            if not improved:
                break
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
