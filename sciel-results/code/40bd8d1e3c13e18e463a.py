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
    
    score = max_conv * n / (total * total * w)
    return score

def _autoconv(heights, M):
    """Compute autoconvolution via FFT."""
    f = np.zeros(M, dtype=np.float64)
    f[:N] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    return conv

def _smooth_max(conv, alpha):
    """Smooth approximation of max(conv) using log-sum-exp."""
    m = np.max(conv)
    if m < 0:
        m = 0.0
    # Clip to avoid overflow
    scaled = alpha * (conv - m)
    scaled = np.clip(scaled, -500, 500)
    exp_conv = np.exp(scaled)
    return m + np.log(np.sum(exp_conv)) / alpha

def _smooth_obj_and_grad(heights, alpha, M):
    """
    Compute smooth objective and its gradient w.r.t. heights.
    Objective: smooth_max(autoconv) / (sum(heights)*w)^2 * (N*w)
    which equals smooth_max * N / (sum^2 * w)
    
    For gradient, we fix the normalization by working with the unnormalized
    smooth_max and then scaling, or we can work with the full expression.
    
    Actually, let's work with: minimize smooth_max(conv) subject to sum(h) = constant.
    But that's a constraint. Instead, let's just minimize the full normalized objective.
    
    Let S = sum(h), w = 0.5/N
    obj = sm * N / (S^2 * w)
    d(obj)/d(h_i) = N/(S^2*w) * d(sm)/d(h_i) + sm * N * d(1/(S^2*w))/d(h_i)
    = N/(S^2*w) * d(sm)/d(h_i) - 2*sm*N/(S^3*w) * 1
    """
    conv = _autoconv(heights, M)
    sm = _smooth_max(conv, alpha)
    
    # Softmax weights for smooth max
    m = np.max(conv)
    if m < 0:
        m = 0.0
    scaled = alpha * (conv - m)
    scaled = np.clip(scaled, -500, 500)
    exp_conv = np.exp(scaled)
    weights = exp_conv / np.sum(exp_conv)
    
    # Gradient of smooth_max w.r.t. heights
    # d(sm)/d(h_i) = sum_k weights[k] * d(conv[k])/d(h_i)
    # d(conv[k])/d(h_i) = 2 * h[k-i] (for linear autoconv, with proper bounds)
    # Use FFT: grad_h = 2 * ifft(rfft(weights) * conj(rfft(h)))[:N]
    W = np.fft.rfft(weights)
    H = np.fft.rfft(heights)
    corr = np.fft.irfft(W * np.conj(H), n=M)
    grad_sm = 2.0 * corr[:N]
    
    S = np.sum(heights)
    w = 0.5 / N
    
    if S <= 1e-10:
        return 1e10, np.zeros(N)
    
    obj = sm * N / (S * S * w)
    
    # Gradient of obj w.r.t. h_i
    # obj = sm * N / (S^2 * w)
    # d(obj)/d(h_i) = N/(S^2*w) * grad_sm[i] + sm * N * (-2*S/(S^4*w)) * 1
    # = N/(S^2*w) * grad_sm[i] - 2*sm*N/(S^3*w)
    grad_obj = (N / (S * S * w)) * grad_sm - (2.0 * sm * N / (S * S * S * w))
    
    return obj, grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    rng = np.random.default_rng(42)
    
    # Generate structured initializations
    candidates = []
    x = np.linspace(0, 1, N)
    
    # Uniform
    candidates.append(np.ones(N, dtype=np.float64))
    
    # Cosine tapers (strong candidates from inspiration)
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
        h_taper = np.ones(N)
        taper_len = int(taper_frac * N / 2)
        if taper_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            h_taper[:taper_len] = ramp
            h_taper[-taper_len:] = ramp[::-1]
        candidates.append(h_taper.copy())
    
    # Parabolic shapes
    for a in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        h_par = 1.0 - a * (2 * x - 1) ** 2
        h_par = np.maximum(h_par, 0.0)
        if np.max(h_par) > 1e-6:
            candidates.append(h_par.copy())
    
    # Gaussian bumps
    for center in [0.3, 0.5, 0.7]:
        for width in [0.1, 0.2, 0.3, 0.4, 0.5]:
            h_gauss = np.exp(-((x - center) / width) ** 2)
            h_gauss = np.maximum(h_gauss, 0.0)
            if np.max(h_gauss) > 1e-6:
                candidates.append(h_gauss.copy())
    
    # Plateau with smooth edges
    for plateau_frac in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        h_plat = np.ones(N)
        edge_len = int((1 - plateau_frac) * N / 2)
        if edge_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge_len)))
            h_plat[:edge_len] = ramp
            h_plat[-edge_len:] = ramp[::-1]
        candidates.append(h_plat.copy())
    
    # Linear tapers
    for slope in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        h_lin = 1.0 - slope * np.abs(2 * x - 1)
        h_lin = np.maximum(h_lin, 0.0)
        if np.max(h_lin) > 1e-6:
            candidates.append(h_lin.copy())
    
    # Evaluate all structured candidates
    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_score(best_h)
    
    for h_cand in candidates:
        if time_left() < 5:
            break
        h_cand = np.maximum(h_cand, 0.0)
        if np.sum(h_cand) <= 1e-10:
            continue
        score = compute_score(h_cand)
        if score < best_score:
            best_score = score
            best_h = h_cand.copy()
    
    # Now do gradient-based optimization from the best candidate
    # Use multiple alpha values and starting points
    alphas = [30.0, 50.0, 100.0, 200.0, 400.0]
    
    # Collect top starting points
    scored_candidates = []
    for h_cand in candidates:
        h_cand = np.maximum(h_cand, 0.0)
        if np.sum(h_cand) <= 1e-10:
            continue
        s = compute_score(h_cand)
        scored_candidates.append((s, h_cand.copy()))
    
    scored_candidates.sort(key=lambda x: x[0])
    top_starts = [h for _, h in scored_candidates[:8]]
    
    # Also add the best_h
    if best_h not in top_starts:
        top_starts.append(best_h.copy())
    
    for alpha in alphas:
        if time_left() < 15:
            break
        
        for h_init in top_starts:
            if time_left() < 8:
                break
            
            h_start = h_init.copy()
            h_start = np.maximum(h_start, 0.0)
            if np.sum(h_start) <= 1e-10:
                continue
            
            try:
                res = minimize(
                    lambda h: _smooth_obj_and_grad(h, alpha, M)[0],
                    h_start,
                    jac=lambda h: _smooth_obj_and_grad(h, alpha, M)[1],
                    method='L-BFGS-B',
                    bounds=[(0.0, None)] * N,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                
                h_res = np.maximum(res.x, 0.0)
                if np.sum(h_res) <= 1e-10:
                    continue
                score = compute_score(h_res)
                if score < best_score:
                    best_score = score
                    best_h = h_res.copy()
            except Exception:
                pass
    
    # Final refinement with highest alpha from best solution
    if time_left() > 10:
        alpha_final = 800.0
        h_start = best_h.copy()
        h_start = np.maximum(h_start, 0.0)
        if np.sum(h_start) > 1e-10:
            try:
                res = minimize(
                    lambda h: _smooth_obj_and_grad(h, alpha_final, M)[0],
                    h_start,
                    jac=lambda h: _smooth_obj_and_grad(h, alpha_final, M)[1],
                    method='L-BFGS-B',
                    bounds=[(0.0, None)] * N,
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
                )
                h_res = np.maximum(res.x, 0.0)
                if np.sum(h_res) > 1e-10:
                    score = compute_score(h_res)
                    if score < best_score:
                        best_score = score
                        best_h = h_res.copy()
            except Exception:
                pass
    
    # Ensure non-negativity
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) > 0:
        best_h /= np.sum(best_h)
    
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
