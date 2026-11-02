import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 135.0  # Safety margin before 150s kill

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
    # Zero-pad to at least 2*n-1 for linear convolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    # Normalization: for uniform heights=1, 
    # integral = n*w = 0.5
    # autoconvolution peak = n*w^2 (for a rectangle of width 0.5, peak of triangle is 0.25)
    # Actually for step function with n intervals of width w, all height 1:
    # The autoconvolution at lag 0 is sum(h_i^2 * w) = n * 1 * w = n*w = 0.5
    # Wait, let me think more carefully.
    # 
    # The step function f(x) = h_i for x in [i*w - 0.25, (i+1)*w - 0.25)
    # Autoconvolution: (f*f)(t) = integral f(x) f(t-x) dx
    # At t=0: sum_i h_i^2 * w = n * 1 * w = 0.5
    # 
    # Normalized max = max(f*f) / (integral(f))^2 * support_length
    # For uniform: 0.5 / (0.5)^2 * 0.5 = 0.5/0.25*0.5 = 1.0
    # 
    # So: score = max_conv / (total * w)^2 * (n * w)
    #       = max_conv * n / (total^2 * w)  ... let me verify
    # integral = total * w
    # score = max_conv / (total*w)^2 * (n*w) = max_conv * n * w / (total^2 * w^2) = max_conv * n / (total^2 * w)
    # For uniform: max_conv = n*w (from discrete sum), total = n
    # score = n*w * n / (n^2 * w) = 1.0 ✓
    
    score = max_conv * n / (total * total * w)
    return score

def _smooth_max(conv, alpha):
    """Smooth approximation of max(conv) using log-sum-exp."""
    m = np.max(conv)
    if m < 0:
        m = 0
    exp_conv = np.exp(alpha * (conv - m))
    return m + np.log(np.sum(exp_conv)) / alpha

def _autoconv_and_grad(heights, M=None):
    """Compute autoconvolution and gradient of max autoconvolution w.r.t. heights."""
    if M is None:
        n = len(heights)
        M = 1
        while M < 2 * n - 1:
            M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:len(heights)] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    peak_idx = np.argmax(conv)
    
    # Gradient of conv[peak_idx] w.r.t. f[i] = 2*f[peak_idx - i] if 0 <= peak_idx - i < len(heights)
    grad = np.zeros(len(heights), dtype=np.float64)
    for i in range(len(heights)):
        j = peak_idx - i
        if 0 <= j < len(heights):
            grad[i] = 2.0 * heights[j]
    
    return conv, grad, peak_idx

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Compute M once for reuse
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    # Strategy: Use smooth optimization with log-sum-exp surrogate
    # The objective is max_t (f*f)(t) / (integral(f))^2 * support
    # We minimize this. Use a smooth surrogate: (1/alpha) * log(sum(exp(alpha * conv_t)))
    # to approximate max(conv_t)
    
    # Also, to handle the normalization, we can fix the integral and just minimize max conv
    # Or use a parameterization that keeps things simple.
    
    # Approach 1: Gradient descent on the smooth surrogate
    # Approach 2: Try structured shapes (tapered, plateau, etc.)
    # Approach 3: Combine both
    
    rng = np.random.default_rng(12345)
    
    # First, try some structured shapes quickly
    structured_candidates = []
    
    # Tapered edges (cosine taper)
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        h_taper = np.ones(N)
        taper_len = int(taper_frac * N)
        if taper_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            h_taper[:taper_len] = ramp
            h_taper[-taper_len:] = ramp[::-1]
        structured_candidates.append(h_taper)
    
    # Raised cosine (smooth bump)
    x = np.linspace(0, 1, N)
    for center in [0.5]:
        for width in [0.3, 0.5, 0.7, 0.9, 1.0]:
            h_cos = np.exp(-((x - center) / width) ** 2)
            h_cos = h_cos / np.max(h_cos)  # normalize to max 1
            structured_candidates.append(h_cos)
    
    # Parabolic
    for a in [0.5, 1.0, 2.0, 3.0]:
        h_par = 1.0 - a * (2 * x - 1) ** 2
        h_par = np.maximum(h_par, 0.0)
        if np.max(h_par) > 0:
            structured_candidates.append(h_par)
    
    # Linear taper
    for slope in [0.5, 1.0, 2.0, 3.0, 5.0]:
        h_lin = 1.0 - slope * np.abs(2 * x - 1)
        h_lin = np.maximum(h_lin, 0.0)
        if np.max(h_lin) > 0:
            structured_candidates.append(h_lin)
    
    # Plateau with smooth edges
    for plateau_frac in [0.3, 0.5, 0.7]:
        h_plat = np.ones(N)
        edge_len = int((1 - plateau_frac) * N / 2)
        if edge_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge_len)))
            h_plat[:edge_len] = ramp
            h_plat[-edge_len:] = ramp[::-1]
        structured_candidates.append(h_plat)
    
    # Evaluate structured candidates
    for h_cand in structured_candidates:
        if time_left() < 5:
            break
        h_cand = np.maximum(h_cand, 0.0)
        if np.sum(h_cand) <= 0:
            continue
        score = compute_score(h_cand)
        if score < best_score:
            best_score = score
            best_h = h_cand.copy()
    
    # Now do gradient-based optimization using scipy L-BFGS-B with smooth surrogate
    
    def smooth_obj_and_grad(heights, alpha):
        """Compute smooth objective and gradient for L-BFGS-B."""
        f = np.zeros(M, dtype=np.float64)
        f[:N] = heights
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        
        sm = _smooth_max(conv, alpha)
        total = np.sum(heights)
        w = 0.5 / N
        if total <= 1e-10:
            return 1e10, np.zeros(N)
        
        # Gradient of smooth_max via log-sum-exp
        m = np.max(conv)
        if m < 0:
            m = 0
        exp_conv = np.exp(alpha * (conv - m))
        weights = exp_conv / np.sum(exp_conv)
        
        # Use FFT for efficient correlation
        W_full = np.fft.rfft(weights)
        F_full = np.fft.rfft(f)
        corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
        grad_f = 2.0 * corr[:N]
        
        S = total
        # obj = sm * N / (S^2 * w)
        obj = sm * N / (S * S * w)
        obj_grad = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
        
        return obj, obj_grad
    
    # Multi-start L-BFGS-B optimization
    alpha_schedule = [30.0, 50.0, 100.0, 200.0, 400.0]
    
    for alpha_val in alpha_schedule:
        if time_left() < 15:
            break
        
        # Start from best so far
        h_opt = best_h.copy()
        
        def objective_wrapper(h):
            obj, _ = smooth_obj_and_grad(h, alpha_val)
            return obj
        
        def gradient_wrapper(h):
            _, grad = smooth_obj_and_grad(h, alpha_val)
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
        
        # Also try a few random restarts at this alpha
        for restart in range(3):
            if time_left() < 10:
                break
            
            freqs = np.arange(N)
            h_init = np.ones(N)
            num_modes = rng.integers(3, 8)
            for _ in range(num_modes):
                k = rng.integers(1, N // 4)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.05, 0.25)
                h_init += amp * np.cos(2 * np.pi * k * freqs / N + phase)
            h_init = np.maximum(h_init, 0.0)
            if np.sum(h_init) <= 0:
                continue
            
            def objective_wrapper_r(h):
                obj, _ = smooth_obj_and_grad(h, alpha_val)
                return obj
            
            def gradient_wrapper_r(h):
                _, grad = smooth_obj_and_grad(h, alpha_val)
                return grad
            
            try:
                result_r = minimize(
                    objective_wrapper_r,
                    h_init,
                    jac=gradient_wrapper_r,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7}
                )
                if result_r.x is not None and np.all(result_r.x >= -1e-10):
                    h_new_r = np.maximum(result_r.x, 0.0)
                    score_r = compute_score(h_new_r)
                    if score_r < best_score:
                        best_score = score_r
                        best_h = h_new_r.copy()
            except Exception:
                pass
    
    # Final polish with high alpha
    if time_left() > 10:
        h_final = best_h.copy()
        alpha_f = 500.0
        
        def objective_final(h):
            obj, _ = smooth_obj_and_grad(h, alpha_f)
            return obj
        
        def gradient_final(h):
            _, grad = smooth_obj_and_grad(h, alpha_f)
            return grad
        
        try:
            result_f = minimize(
                objective_final,
                h_final,
                jac=gradient_final,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result_f.x is not None and np.all(result_f.x >= -1e-10):
                h_new_f = np.maximum(result_f.x, 0.0)
                score_f = compute_score(h_new_f)
                if score_f < best_score:
                    best_score = score_f
                    best_h = h_new_f.copy()
        except Exception:
            pass
    
    # Ensure non-negativity
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
