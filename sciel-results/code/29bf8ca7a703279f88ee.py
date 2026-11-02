import numpy as np
import time
from scipy.optimize import minimize

N = 600
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

# Precompute FFT setup
K = 16
M_STEP = N * K  # 9600
M_FFT = 1 << 14  # 16384

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    
    if integral_f <= 0:
        return np.inf
    
    score = max_conv / (integral_f ** 2)
    return score

def _autoconv_and_grad(h, beta=40.0):
    """
    Compute smooth-max surrogate of normalized autoconvolution peak and its gradient.
    h: array of N nonnegative step heights
    Returns (objective_value, gradient_wrt_h)
    """
    h = np.maximum(h, 0.0)
    n = len(h)
    w = 0.5 / n
    
    # Build fine grid
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(h, K)
    
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    integral_f = np.sum(h) * w
    if integral_f <= 0:
        return 1e10, np.zeros_like(h)
    
    # The autoconvolution of a function on [-1/4,1/4] has support on [-1/2,1/2]
    # In the FFT representation, lag 0 is at index 0
    # We need to consider the circular convolution carefully
    # For a function supported on [0, 0.5] in the FFT grid of size M_FFT,
    # the linear autoconvolution peak is at lag 0 (index 0)
    
    # Normalize
    norm_conv = conv / (integral_f ** 2)
    
    # Use a window around where the peak is likely to be
    # The peak of autoconvolution for nonnegative f is at lag 0
    # In our FFT setup, index 0 corresponds to lag 0
    # But we should check a window around the center as well due to circular effects
    center = M_FFT // 2
    half_win = M_FFT // 4
    start_idx = max(0, center - half_win)
    end_idx = min(M_FFT, center + half_win)
    conv_window = norm_conv[start_idx:end_idx]
    
    # Also include the beginning (lag 0 region)
    conv_window = np.concatenate([norm_conv[:half_win], conv_window])
    
    # Smooth max using log-sum-exp
    beta_safe = max(beta, 1.0)
    max_val = np.max(conv_window)
    exp_terms = np.exp(beta_safe * (conv_window - max_val))
    log_sum = np.log(np.sum(exp_terms))
    smooth_max_val = max_val + log_sum / beta_safe
    
    # Gradient of smooth max w.r.t. conv_window
    sum_exp = np.sum(exp_terms)
    grad_conv_window = exp_terms / sum_exp
    
    # Gradient w.r.t. conv
    grad_conv = np.zeros(M_FFT)
    # First half comes from the beginning
    first_len = half_win
    grad_conv[:first_len] = grad_conv_window[:first_len]
    grad_conv[start_idx:end_idx] = grad_conv_window[first_len:]
    
    # Gradient w.r.t. f_fine using the adjoint
    # conv = ifft(F * F) where F = fft(f_fine)
    # d(conv)/d(f_fine) via Parseval:
    # grad_f_fine = ifft(2 * conj(F) * fft(grad_conv))
    G = np.fft.rfft(grad_conv)
    grad_f_fine = np.fft.irfft(2 * np.conj(F) * G, n=M_FFT)
    
    # Aggregate from fine grid to step heights
    grad_h = np.zeros(N)
    for i in range(N):
        start = i * K
        end = start + K
        if end <= M_STEP:
            grad_h[i] = np.sum(grad_f_fine[start:end])
    
    # Gradient of normalization
    # score = max_conv / integral_f^2
    # d(score)/d(h) = d(max_conv)/d(h) / integral_f^2 - 2*max_conv*d(integral_f)/d(h) / integral_f^3
    d_int = w * np.ones(N)
    grad_h = grad_h / (integral_f ** 2) - 2 * smooth_max_val * d_int / (integral_f ** 3)
    
    return smooth_max_val, grad_h

def _two_block_optimize(h0, max_iter=50, beta=40.0, time_budget=20.0):
    """
    Optimize a two-block structure: h[i] = a for i in [0, n1), b for i in [n1, N)
    This is a fast way to find good initial structures.
    """
    start_t = time.time()
    
    # Define the two-block parametrization
    def two_block_score(params):
        a, b, n1_frac = params
        n1 = int(n1_frac * N)
        h = np.zeros(N)
        h[:n1] = a
        h[n1:] = b
        h = np.maximum(h, 0)
        return compute_score(h)
    
    # Try a grid of two-block configurations
    best_params = None
    best_score = compute_score(h0)
    best_h = h0.copy()
    
    for n1_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for a in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]:
            for b in [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]:
                h = np.zeros(N)
                n1 = int(n1_frac * N)
                h[:n1] = a
                h[n1:] = b
                h = np.maximum(h, 0)
                if np.sum(h) <= 0:
                    continue
                score = compute_score(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
                    best_params = (a, b, n1_frac)
        
        if time.time() - start_t > time_budget:
            break
    
    # Refine with local search around best params
    if best_params is not None:
        a, b, n1_frac = best_params
        # Coarse local search
        for da in [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]:
            for db in [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]:
                for dn1 in [-0.05, -0.02, 0, 0.02, 0.05]:
                    a2 = max(0, a + da)
                    b2 = max(0, b + db)
                    n1_frac2 = min(1.0, max(0.05, n1_frac + dn1))
                    h = np.zeros(N)
                    n1 = int(n1_frac2 * N)
                    h[:n1] = a2
                    h[n1:] = b2
                    h = np.maximum(h, 0)
                    if np.sum(h) <= 0:
                        continue
                    score = compute_score(h)
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()
        
        if time.time() - start_t > time_budget:
            break
    
    return best_h, best_score

def _lbfgs_optimize(h0, max_iter=80, beta=40.0, time_budget=30.0):
    """Optimize using L-BFGS-B with smooth-max objective."""
    start_t = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_t > time_budget:
            return 1e10, np.zeros_like(x)
        val, grad = _autoconv_and_grad(x, beta=beta)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(x)
        return val, grad
    
    bounds = [(0, None)] * N
    
    try:
        result = minimize(
            obj_and_grad,
            h0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 3,
                'ftol': 1e-7,
                'gtol': 1e-5,
                'disp': False
            }
        )
    except Exception:
        return h0, compute_score(h0)
    
    h_opt = np.maximum(result.x, 0.0)
    score = compute_score(h_opt)
    
    return h_opt, score

def _coordinate_descent(h, max_rounds=10, time_budget=15.0):
    """
    Simple coordinate descent: adjust individual heights one at a time.
    """
    start_t = time.time()
    h = h.copy()
    
    for round_num in range(max_rounds):
        if time.time() - start_t > time_budget:
            break
        improved = False
        for i in range(N):
            if time.time() - start_t > time_budget - 1:
                break
            old_val = h[i]
            # Try a few values
            for delta in [-0.1, -0.05, 0.05, 0.1]:
                h[i] = max(0, old_val + delta)
                score = compute_score(h)
                if score < compute_score(np.concatenate([h[:i], h[i:], h[i+1:]])) if False else True:
                    # Just compare with current best
                    pass
            # Simpler: try a small set of values
            for val in [max(0, old_val - 0.1), max(0, old_val - 0.03), old_val, old_val + 0.03, old_val + 0.1]:
                h[i] = val
                score = compute_score(h)
                if score < compute_score(h):
                    improved = True
                    break
            # Reset to best found
            h[i] = old_val
            for val in [max(0, old_val - 0.1), max(0, old_val - 0.03), old_val, old_val + 0.03, old_val + 0.1]:
                h[i] = val
                s_new = compute_score(h)
                s_old = compute_score(np.array([h[:i], h[i:], h[i+1:]]).reshape(N)) if i < N-1 else compute_score(h)
                if s_new < best_score_global:
                    best_score_global = s_new
                    h[i] = val
                    improved = True
                    break
        if not improved:
            break
    
    return h

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    x = np.linspace(0, 1, N)
    
    # Strategy 1: Two-block optimization (fast, effective)
    h2, score2 = _two_block_optimize(h, time_budget=20.0)
    if score2 < best_score:
        best_score = score2
        best_h = h2.copy()
    
    # Strategy 2: Try various smooth shapes
    for amp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        h_cos = 1.0 - amp * 0.5 * (1 - np.cos(2 * np.pi * x))
        h_cos = np.maximum(h_cos, 0.0)
        score = compute_score(h_cos)
        if score < best_score:
            best_score = score
            best_h = h_cos.copy()
    
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        h_plateau = np.ones(N)
        taper_len = int(taper_frac * N)
        if taper_len > 0:
            ramp = np.linspace(0, 1, taper_len)
            h_plateau[:taper_len] = ramp
            h_plateau[-taper_len:] = ramp[::-1]
        score = compute_score(h_plateau)
        if score < best_score:
            best_score = score
            best_h = h_plateau.copy()
    
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        h_gauss = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
        h_gauss = h_gauss / np.max(h_gauss)
        score = compute_score(h_gauss)
        if score < best_score:
            best_score = score
            best_h = h_gauss.copy()
    
    # Strategy 3: L-BFGS-B from multiple starting points
    start_points = [best_h.copy()]
    
    # Add a few diverse starting points
    for taper_frac in [0.15, 0.25, 0.35]:
        h_plateau = np.ones(N)
        taper_len = int(taper_frac * N)
        if taper_len > 0:
            ramp = np.linspace(0, 1, taper_len)
            h_plateau[:taper_len] = ramp
            h_plateau[-taper_len:] = ramp[::-1]
        start_points.append(h_plateau)
    
    remaining_time = TIME_LIMIT - (time.time() - START_TIME)
    num_starts = min(len(start_points), 4)
    time_per_start = max(8, (remaining_time - 10) / max(num_starts, 1))
    
    for i, h_start in enumerate(start_points[:num_starts]):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        time_budget = min(time_per_start, TIME_LIMIT - (time.time() - START_TIME) - 8)
        if time_budget < 5:
            break
        
        h_opt, score = _lbfgs_optimize(h_start, max_iter=60, beta=40.0, time_budget=time_budget)
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Strategy 4: Final local polish with small perturbations
    if time.time() - START_TIME < TIME_LIMIT - 5:
        rng = np.random.default_rng(42)
        for _ in range(30):
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            perturbation = rng.normal(0, 0.02, N)
            h_trial = best_h + perturbation
            h_trial = np.maximum(h_trial, 0.0)
            score = compute_score(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
