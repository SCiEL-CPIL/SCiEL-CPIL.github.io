import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 138.0  # Safety margin before 150s kill
M_FFT = 2048  # Next power of 2 >= 2*600-1 = 1199

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
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

def _autoconv(heights):
    """Compute autoconvolution using FFT."""
    n = len(heights)
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    return conv

def _smooth_obj_and_grad(heights, alpha):
    """Compute smooth objective and gradient for L-BFGS-B.
    
    Uses log-sum-exp for smooth max approximation.
    """
    n = len(heights)
    f = np.zeros(M_FFT, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    # Log-sum-exp for smooth max
    max_c = np.max(conv)
    shifted = conv - max_c
    exp_conv = np.exp(alpha * shifted)
    sum_exp = np.sum(exp_conv)
    smooth_max = max_c + np.log(sum_exp) / alpha
    
    # Gradient weights
    weights = exp_conv / sum_exp
    
    # Gradient of smooth_max w.r.t. heights
    # d(smooth_max)/d(h_i) = sum_j weights[j] * d(conv[j])/d(h_i) / 1
    # d(conv[j])/d(h_i) = h[j-i] + h[j+i] (with zero padding)
    
    # Use FFT for efficient computation
    W_full = np.fft.rfft(weights)
    F_full = np.fft.rfft(f)
    
    # Correlation: sum_j weights[j] * h[j-i]
    # This is ifft(fft(weights) * conj(fft(h)))
    corr = np.fft.irfft(W_full * np.conj(F_full), n=M_FFT)
    
    # The gradient is 2 * corr[:n] (because of the two terms h[j-i] + h[j+i])
    # Actually, let's be more careful.
    # d(conv[j])/d(h[i]) = h[j-i] + h[j+i]
    # grad[i] = sum_j weights[j] * (h[j-i] + h[j+i])
    #         = sum_j weights[j]*h[j-i] + sum_j weights[j]*h[j+i]
    
    # First term: sum_j weights[j]*h[j-i] is correlation at lag i
    # Second term: sum_j weights[j]*h[j+i] is correlation at lag -i
    
    # Using FFT:
    # corr1[i] = sum_j weights[j] * h[j-i]
    # This can be computed as ifft(fft(weights) * conj(fft(h)))
    
    # Let's compute both terms
    W = np.fft.rfft(weights)
    H = np.fft.rfft(f)
    
    # Term 1: sum_j weights[j] * h[j-i]
    # = ifft(W * conj(H))
    term1 = np.fft.irfft(W * np.conj(H), n=M_FFT)
    
    # Term 2: sum_j weights[j] * h[j+i]
    # = sum_j weights[j] * h[j+i]
    # Let k = j+i, then j = k-i
    # = sum_k weights[k-i] * h[k]
    # This is correlation of weights and h, but with different indexing
    # = ifft(conj(W) * H)  -- actually this is the same as term1 by symmetry
    # Wait, let's think again.
    
    # Actually, for real signals, the correlation is symmetric.
    # sum_j w[j]*h[j-i] and sum_j w[j]*h[j+i] are related.
    
    # Let's just compute the gradient directly using the formula:
    # grad[i] = (1/alpha) * sum_j exp(alpha*(conv[j]-max_c)) * (h[j-i] + h[j+i])
    
    # Since we already have weights = exp(alpha*(conv-max_c)) / sum_exp,
    # grad[i] = (1/alpha) * sum_j weights[j] * sum_exp * (h[j-i] + h[j+i])
    #         = sum_j weights[j] * (h[j-i] + h[j+i])
    
    # So grad = term1 + term2 where:
    # term1[i] = sum_j weights[j] * h[j-i]
    # term2[i] = sum_j weights[j] * h[j+i]
    
    # term1 is the correlation of weights and h
    # term2 is also a correlation but with different sign
    
    # For real-valued signals, if we define:
    # corr[i] = sum_j weights[j] * h[j-i]
    # then corr[-i] = sum_j weights[j] * h[j+i]
    
    # So grad[i] = corr[i] + corr[-i]
    
    # But since our arrays are indexed from 0 to M_FFT-1, we need to be careful.
    
    # Let's just compute it numerically for correctness, since N=600 is small
    # Actually, let's use the FFT approach more carefully.
    
    # The autoconvolution is conv = h * h (linear convolution)
    # d(conv)/dh = 2 * h (in the sense that d(conv[j])/dh[i] = h[j-i] + h[j+i])
    
    # So if we define L = sum_j weights[j] * conv[j], then
    # dL/dh[i] = sum_j weights[j] * (h[j-i] + h[j+i])
    
    # This is equivalent to:
    # dL/dh = 2 * (weights * h) where * denotes appropriate cross-correlation
    
    # Using FFT:
    # Let W = fft(weights), H = fft(h)
    # Then the cross-correlation can be computed as ifft(W * conj(H))
    
    # But we need to handle the two terms separately.
    
    # Actually, the cleanest way:
    # dL/dh[i] = sum_j weights[j] * h[j-i] + sum_j weights[j] * h[j+i]
    
    # First sum: let k = j-i, then j = k+i
    # = sum_k weights[k+i] * h[k]
    # This is the cross-correlation of weights and h at lag i
    
    # Second sum: let k = j+i, then j = k-i
    # = sum_k weights[k-i] * h[k]
    # This is the cross-correlation of weights and h at lag -i
    
    # So dL/dh[i] = corr(i) + corr(-i)
    
    # Using FFT, we can compute corr(i) for all i at once.
    
    # Let's compute it using the following approach:
    # Create an extended array for weights and h
    # Then use ifft to get the correlation
    
    # For now, let's use a direct computation since it's not too slow
    # grad = np.zeros(n)
    # for i in range(n):
    #     for j in range(M_FFT):
    #         if 0 <= j-i < n:
    #             grad[i] += weights[j] * heights[j-i]
    #         if 0 <= j+i < n:
    #             grad[i] += weights[j] * heights[j+i]
    
    # But that's O(N*M) = 600*2048 = 1.2M operations, which is fine.
    
    # Let's use a vectorized approach:
    grad = np.zeros(n)
    
    for j in range(M_FFT):
        wj = weights[j]
        if wj < 1e-15:
            continue
        
        # h[j-i] for i in range(n): j-i in [0, n-1] => i in [j-n+1, j]
        i_min = max(0, j - n + 1)
        i_max = min(n - 1, j)
        if i_min <= i_max:
            idx = j - np.arange(i_min, i_max + 1)
            grad[i_min:i_max+1] += wj * heights[idx]
        
        # h[j+i] for i in range(n): j+i in [0, n-1] => i in [-j, n-1-j]
        i_min2 = max(0, -j)
        i_max2 = min(n - 1, n - 1 - j)
        if i_min2 <= i_max2:
            idx2 = j + np.arange(i_min2, i_max2 + 1)
            grad[i_min2:i_max2+1] += wj * heights[idx2]
    
    # Now compute the objective and its gradient
    total = np.sum(heights)
    w = 0.5 / n
    
    if total <= 1e-10:
        return 1e10, np.zeros(n)
    
    # obj = smooth_max * n / (total^2 * w)
    obj = smooth_max * n / (total * total * w)
    
    # d(obj)/d(h_i) = d(smooth_max)/d(h_i) * n / (total^2 * w) 
    #                + smooth_max * n * d(1/(total^2 * w))/d(h_i)
    #                = grad[i] * n / (total^2 * w)
    #                + smooth_max * n * (-2 * total * w) / (total^4 * w^2) * dh_i/dh_i
    #                = grad[i] * n / (total^2 * w)
    #                - 2 * smooth_max * n / (total^3 * w)
    
    obj_grad = grad * (n / (total * total * w)) - smooth_max * (2.0 * n / (total * total * total * w))
    
    return obj, obj_grad

def _parametric_heights(params):
    """Generate heights from parametric representation.
    
    params: [center, width, taper_power, amplitude]
    """
    center = params[0]
    width = params[1]
    taper_power = params[2]
    amplitude = params[3]
    
    x = np.linspace(0, 1, N)
    d = np.abs(x - center)
    
    ratio = d / max(width, 1e-10)
    mask = ratio < 1.0
    heights = np.zeros(N)
    
    p = max(taper_power, 0.1)
    heights[mask] = amplitude * (1.0 - ratio[mask] ** p) ** (1.0 / p)
    
    return np.maximum(heights, 0.0)

def _parametric_score(params):
    """Score for parametric representation."""
    h = _parametric_heights(params)
    if np.sum(h) <= 0:
        return 1e10
    return compute_score(h)

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0
    
    rng = np.random.default_rng(42)
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N, dtype=np.float64))
    
    # 2. Gaussian centered
    x = np.linspace(0, 1, N)
    for width in [0.1, 0.2, 0.3, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
    
    # 3. Two blocks
    for left_frac in [0.2, 0.3, 0.4]:
        for gap_frac in [0.0, 0.1, 0.2]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                candidates.append(hb)
                
    # 4. Random sparse
    for _ in range(5):
        hr = np.zeros(N)
        mask = rng.random(N) < 0.5
        hr[mask] = 1.0
        candidates.append(hr)
        
    # 5. Cosine tapers
    for tf in [0.1, 0.2, 0.3, 0.5]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        candidates.append(ht)

    # Evaluate initial candidates
    for hc in candidates:
        if deadline_reached():
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()

    # Strategy 1: Parametric search with grid + Nelder-Mead
    if time_left() > 30:
        center_vals = np.linspace(0.3, 0.7, 5)
        width_vals = np.linspace(0.3, 1.0, 6)
        taper_vals = [1.0, 2.0, 4.0, 8.0, 16.0]
        
        best_params = None
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
                    for amp in [0.5, 1.0, 1.5]:
                        params = [c, wdt, tp, amp]
                        score = _parametric_score(params)
                        if score < best_param_score:
                            best_param_score = score
                            best_params = params
                            h_new = _parametric_heights(params)
                            best_score = score
                            best_h = h_new.copy()
        
        # Nelder-Mead refinement
        if best_params is not None and time_left() > 15:
            param_bounds = [
                (0.2, 0.8),
                (0.1, 1.0),
                (0.5, 20.0),
                (0.1, 2.0)
            ]
            
            def param_obj(p):
                return _parametric_score(p)
            
            try:
                result = minimize(
                    param_obj,
                    best_params,
                    method='Nelder-Mead',
                    bounds=param_bounds,
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.x is not None:
                    h_new = _parametric_heights(result.x)
                    score = compute_score(h_new)
                    if score < best_score:
                        best_score = score
                        best_h = h_new.copy()
            except Exception:
                pass
    
    # Strategy 2: Multi-start Simulated Annealing
    if time_left() > 40:
        num_chains = 3
        steps_per_chain = 1200
        
        for chain_idx in range(num_chains):
            if deadline_reached():
                break
            
            if chain_idx == 0:
                current_h = best_h.copy()
            else:
                if rng.random() < 0.5:
                    current_h = candidates[rng.integers(0, len(candidates))].copy()
                else:
                    current_h = np.maximum(np.ones(N) + rng.normal(0, 0.5, N), 0.0)
            
            current_score = compute_score(current_h)
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
            
            T_start = 0.1
            T_end = 0.001
            step_size = 0.1
            
            for step in range(steps_per_chain):
                if deadline_reached():
                    break
                
                progress = step / steps_per_chain
                T = T_start * (T_end / T_start) ** progress
                
                noise = rng.normal(0, step_size, N)
                new_h = current_h + noise
                new_h = np.maximum(new_h, 0.0)
                
                if np.sum(new_h) <= 0:
                    continue
                    
                new_score = compute_score(new_h)
                
                if new_score < current_score:
                    accept = True
                else:
                    delta = new_score - current_score
                    if delta < 0.05: 
                        accept = rng.random() < np.exp(-delta / T)
                    else:
                        accept = False
                
                if accept:
                    current_h = new_h
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
                    step_size = max(0.01, step_size * 0.999)
                else:
                    step_size = max(0.01, step_size * 0.95)
    
    # Strategy 3: Gradient-based optimization with smooth surrogate (L-BFGS-B)
    if time_left() > 20:
        alpha_schedule = [20.0, 50.0, 100.0, 200.0]
        
        for alpha_val in alpha_schedule:
            if deadline_reached():
                break
            
            h_opt = best_h.copy()
            
            def objective_wrapper(h, alpha=alpha_val):
                obj, _ = _smooth_obj_and_grad(h, alpha)
                return obj
            
            def gradient_wrapper(h, alpha=alpha_val):
                _, grad = _smooth_obj_and_grad(h, alpha)
                return grad
            
            try:
                result = minimize(
                    objective_wrapper,
                    h_opt,
                    jac=gradient_wrapper,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.x is not None and np.all(result.x >= -1e-10):
                    h_new = np.maximum(result.x, 0.0)
                    score = compute_score(h_new)
                    if score < best_score:
                        best_score = score
                        best_h = h_new.copy()
            except Exception:
                pass
            
            # Random restarts
            for restart in range(3):
                if deadline_reached():
                    break
                
                h_init = np.maximum(np.ones(N) + rng.normal(0, 0.3, N), 0.0)
                
                try:
                    result = minimize(
                        objective_wrapper,
                        h_init,
                        jac=gradient_wrapper,
                        method='L-BFGS-B',
                        bounds=[(0, None)] * N,
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if result.x is not None and np.all(result.x >= -1e-10):
                        h_new = np.maximum(result.x, 0.0)
                        score = compute_score(h_new)
                        if score < best_score:
                            best_score = score
                            best_h = h_new.copy()
                except Exception:
                    pass
    
    # Final polish: Local random search with adaptive step size
    if time_left() > 10:
        polish_steps = 200
        current_h = best_h.copy()
        current_score = best_score
        step_size = 0.05
        
        for _ in range(polish_steps):
            if deadline_reached():
                break
            noise = rng.normal(0, step_size, N)
            new_h = np.maximum(current_h + noise, 0.0)
            new_score = compute_score(new_h)
            
            if new_score < current_score:
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                step_size = min(0.1, step_size * 1.05)
            else:
                step_size = max(0.005, step_size * 0.9)

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
