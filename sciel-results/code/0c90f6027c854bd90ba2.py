import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, minimize_scalar

N = 50
w = 1.0 / 100.0
m = 2 * N - 1  # 99
m_plus_1 = m + 1  # 100

# Precompute FFT size
_N_FFT = 1
while _N_FFT < 2 * N - 1:
    _N_FFT <<= 1

# Precompute DCT matrices
from scipy.fft import dct, idct

# DCT-II matrix for N points
def _dct_matrix(n):
    """Return DCT-II matrix of size n x n."""
    k = np.arange(n)
    m_idx = np.arange(n)
    M = np.cos(np.pi * (2*k[:, None] + 1) * m_idx[None, :] / (2*n))
    return M

_DCT = _dct_matrix(N)
_IDCT = _DCT.T  # DCT-II is orthogonal up to scaling, use transpose for inverse

def compute_value(h):
    """
    Compute the objective value for given step heights h.
    h: array of N nonnegative numbers.
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    
    # FFT based convolution
    H = np.fft.rfft(h, _N_FFT)
    G = H * H
    g = np.fft.irfft(G, _N_FFT)[:m]
    g = np.real(g)
    
    # Clip small negative values from numerical noise
    g = np.maximum(g, 0.0)
    
    # Calculate components
    sum_g_sq = np.sum(g**2)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    sum_g = np.sum(g)
    max_g = np.max(g)
    
    if max_g == 0 or sum_g == 0:
        return 0.0
        
    # Value = (w/3 * (sum_g_sq + 2*sum_g_g_next)) / ( (sum_g/(m+1)) * max_g )
    # = (w * (m+1) * (sum_g_sq + 2*sum_g_g_next)) / (3 * sum_g * max_g)
    
    numerator = w * m_plus_1 * (sum_g_sq + 2.0 * sum_g_g_next)
    denominator = 3.0 * sum_g * max_g
    
    return numerator / denominator

def compute_value_smooth(h, smoothing=1e-4):
    """
    Compute the objective value with smoothed max for better gradient flow.
    Uses log-sum-exp based soft maximum for norm_inf.
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    
    # FFT based convolution
    H = np.fft.rfft(h, _N_FFT)
    G = H * H
    g = np.fft.irfft(G, _N_FFT)[:m]
    g = np.real(g)
    
    # Clip small negative values from numerical noise
    g = np.maximum(g, 0.0)
    
    # Calculate components
    sum_g_sq = np.sum(g**2)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    sum_g = np.sum(g)
    max_g = np.max(g)
    
    if max_g == 0 or sum_g == 0:
        return 0.0
    
    # Smooth max using log-sum-exp: soft_max = max + smoothing * log(sum(exp((g-max)/smoothing)))
    shifted = (g - max_g) / smoothing
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -50, 50)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    smooth_max = max_g + smoothing * log_sum_exp
    
    numerator = w * m_plus_1 * (sum_g_sq + 2.0 * sum_g_g_next)
    denominator = 3.0 * sum_g * smooth_max
    
    return numerator / denominator

def neg_value(log_h):
    """
    Objective function for minimization in log-space (smooth version).
    """
    # Center to prevent overflow/underflow issues in exp
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value_smooth(h, smoothing=1e-4)

def neg_value_exact(log_h):
    """
    Objective function for minimization in log-space (exact version).
    """
    # Center to prevent overflow/underflow issues in exp
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value(h)

def parametric_h_gaussian(params):
    """
    h_i = exp(-(i-c)^2 / (2*sigma^2))
    params: [log_sigma]
    """
    sigma = np.exp(params[0])
    x = np.arange(N)
    c = (N - 1) / 2.0
    h = np.exp(-((x - c)**2) / (2.0 * sigma**2))
    return h

def parametric_h_dct_spectral(params):
    """
    h_i = softplus(DCT^{-1}(exp(spectral_coeffs)))
    params: [log_scale, *spectral_coeffs (first K)]
    Uses DCT to represent h in spectral domain, ensuring smooth shapes.
    softplus(x) = log(1 + exp(x)) ensures h >= 0.
    """
    log_scale = params[0]
    scale = np.exp(log_scale)
    
    K = min(15, N)  # Use first K spectral coefficients
    spectral = params[1:1+K]
    
    # Build full spectral vector (zero-pad the rest)
    full_spec = np.zeros(N)
    full_spec[:K] = spectral
    
    # Inverse DCT to get h in spatial domain
    h_raw = _IDCT @ full_spec * scale
    
    # Apply softplus to ensure non-negativity
    # Use stable softplus: log(1+exp(x)) = max(x,0) + log1p(exp(-|x|))
    h = np.where(h_raw > 20, h_raw, np.log1p(np.exp(h_raw)))
    
    # Ensure finite and non-negative
    h = np.maximum(h, 0.0)
    h = np.where(np.isfinite(h), h, 0.0)
    
    return h

def parametric_h_super_gaussian(params):
    """
    h_i = exp(-(|i-c|^beta) / (2*sigma^beta))
    params: [log_sigma, beta]
    """
    sigma = np.exp(params[0])
    beta = params[1]
    x = np.arange(N)
    c = (N - 1) / 2.0
    dist = np.abs(x - c)
    h = np.exp(-(dist**beta) / (2.0 * (sigma**beta)))
    return h

def parametric_h_polynomial(params):
    """
    h_i = (1 - (i-c)^2 / R^2)_+^p
    params: [log_R, p]
    """
    R = np.exp(params[0])
    p = params[1]
    x = np.arange(N)
    c = (N - 1) / 2.0
    t = (x - c) / R
    h = np.maximum(1.0 - t**2, 0.0)
    h = h**p
    return h

def parametric_h_flat_top(params):
    """
    Flat top with cosine edges.
    h_i = 1 for |i-c| < w/2, cosine falloff to 0 at |i-c| = w.
    params: [log_w]
    """
    w_width = np.exp(params[0])
    x = np.arange(N)
    c = (N - 1) / 2.0
    d = np.abs(x - c)
    half_w = w_width / 2.0
    h = np.zeros(N)
    mask_inner = d < half_w
    mask_edge = (d >= half_w) & (d <= w_width)
    h[mask_inner] = 1.0
    # Cosine falloff: 0.5 * (1 + cos(pi * (d - half_w) / (w_width - half_w)))
    # At d=half_w, arg=0, cos=1, val=1.
    # At d=w_width, arg=pi, cos=-1, val=0.
    d_edge = d[mask_edge]
    t = (d_edge - half_w) / (w_width - half_w)
    h[mask_edge] = 0.5 * (1.0 + np.cos(np.pi * t))
    return h

def parametric_h_hamming(params):
    """
    Hamming window: h_i = 0.54 - 0.46*cos(2*pi*(i-c)/W) for |i-c| <= W/2, else 0.
    params: [log_W]
    """
    W = np.exp(params[0])
    x = np.arange(N)
    c = (N - 1) / 2.0
    d = np.abs(x - c)
    h = np.zeros(N)
    mask = d <= W / 2.0
    t = 2.0 * np.pi * (x[mask] - c) / W
    h[mask] = 0.54 - 0.46 * np.cos(t)
    h = np.maximum(h, 0.0)
    return h

def parametric_h_blackman(params):
    """
    Blackman window: h_i = 0.42 - 0.5*cos(2*pi*(i-c)/W) + 0.08*cos(4*pi*(i-c)/W) for |i-c| <= W/2, else 0.
    params: [log_W]
    """
    W = np.exp(params[0])
    x = np.arange(N)
    c = (N - 1) / 2.0
    d = np.abs(x - c)
    h = np.zeros(N)
    mask = d <= W / 2.0
    t = 2.0 * np.pi * (x[mask] - c) / W
    h[mask] = 0.42 - 0.5 * np.cos(t) + 0.08 * np.cos(2.0 * t)
    h = np.maximum(h, 0.0)
    return h

def parametric_h_two_gauss(params):
    """
    Sum of two Gaussians.
    params: [log_sigma1, log_sigma2, offset, amplitude_ratio]
    """
    sigma1 = np.exp(params[0])
    sigma2 = np.exp(params[1])
    offset = params[2]
    amp_ratio = params[3]
    x = np.arange(N)
    c = (N - 1) / 2.0
    h = np.exp(-((x - (c - offset))**2) / (2.0 * sigma1**2))
    h += amp_ratio * np.exp(-((x - (c + offset))**2) / (2.0 * sigma2**2))
    return h

def optimize_parametric(func, bounds, initial_guess=None, maxiter=100, tol=1e-6):
    """
    Use differential_evolution for global optimization of parameters.
    """
    def obj(params):
        try:
            h = func(params)
            if np.any(h < 0) or np.any(~np.isfinite(h)) or np.max(h) == 0:
                return 1e6
            return -compute_value(h)
        except:
            return 1e6
            
    try:
        res = differential_evolution(obj, bounds, seed=42, maxiter=maxiter, tol=tol, polish=True)
        best_params = res.x
        best_h = func(best_params)
        best_val = compute_value(best_h)
        return best_val, best_h, best_params
    except Exception:
        return 0.0, None, None

def optimize_dct_spectral(time_budget=30.0):
    """
    Optimize DCT spectral coefficients using a combination of global and local search.
    """
    start_time = time.time()
    best_val = 0.0
    best_h = np.ones(N)
    
    # Start with uniform (all DCT coefficients zero except DC)
    K = 10  # Number of spectral coefficients to optimize
    bounds = [[0.0, 2.0]]  # log_scale
    for _ in range(K):
        bounds.append([-3.0, 3.0])  # spectral coefficients
    
    # Global search with reduced iterations
    val, h, params = optimize_parametric(parametric_h_dct_spectral, bounds, maxiter=80, tol=1e-5)
    if h is not None and val > best_val:
        best_val = val
        best_h = h.copy()
    
    # Local refinement in log-space
    if time.time() - start_time < time_budget:
        log_h_start = np.log(np.maximum(best_h, 1e-6))
        try:
            res = minimize(
                neg_value,
                log_h_start,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'maxfun': 8000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_refined = np.exp(res.x - np.mean(res.x))
            h_refined = np.maximum(h_refined, 0.0)
            val_refined = compute_value(h_refined)
            if val_refined > best_val:
                best_val = val_refined
                best_h = h_refined.copy()
        except Exception:
            pass
    
    return best_val, best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Store best solution found
    best_val = 0.0
    best_h = np.ones(N)
    seed_val = compute_value(best_h)
    if seed_val > best_val:
        best_val = seed_val
        best_h = best_h.copy()
        
    # List of candidate h vectors for local refinement
    candidates = []
    
    # 1. Seed
    candidates.append(np.ones(N))
    
    # 2. DCT Spectral Optimization (primary strategy)
    if time.time() < deadline - 60.0:
        dct_val, dct_h = optimize_dct_spectral(time_budget=50.0)
        if dct_h is not None and dct_val > best_val:
            best_val = dct_val
            best_h = dct_h.copy()
        if dct_h is not None:
            candidates.append(dct_h)
    
    # 3. Quick parametric global optimization (reduced iterations)
    families = [
        ("gauss", parametric_h_gaussian, [[0.0, 3.0]]),
        ("super_gauss", parametric_h_super_gaussian, [[0.0, 3.0], [1.0, 5.0]]),
        ("two_gauss", parametric_h_two_gauss, [[0.0, 3.0], [0.0, 3.0], [0.0, 10.0], [0.1, 2.0]]),
    ]
    
    for name, func, bounds in families:
        if time.time() > deadline - 40.0:
            break
        val, h, _ = optimize_parametric(func, bounds, maxiter=60, tol=1e-5)
        if h is not None and val > best_val:
            best_val = val
            best_h = h.copy()
        if h is not None:
            candidates.append(h)
            
    # 4. Add heuristic candidates
    x = np.arange(N)
    c = (N - 1) / 2.0
    # Triangle
    h_tri = np.maximum(1.0 - np.abs(x - c) / c, 0.0)
    candidates.append(h_tri)
    # Raised Cosine
    h_rc = 0.5 * (1.0 + np.cos(np.pi * (x - c) / c))
    candidates.append(h_rc)
    # Two bumps
    h_2b = np.exp(-((x - (c-5))**2)/5) + np.exp(-((x - (c+5))**2)/5)
    candidates.append(h_2b)
    # Narrow Gaussian
    h_narrow = np.exp(-((x - c)**2) / (2.0 * 3.0**2))
    candidates.append(h_narrow)
    # Wide Gaussian
    h_wide = np.exp(-((x - c)**2) / (2.0 * 12.0**2))
    candidates.append(h_wide)
    # Hamming-like
    h_ham = 0.54 - 0.46 * np.cos(2.0 * np.pi * (x - c) / (N - 1))
    h_ham = np.maximum(h_ham, 0.0)
    candidates.append(h_ham)
    # Blackman-like
    t = 2.0 * np.pi * (x - c) / (N - 1)
    h_blk = 0.42 - 0.5 * np.cos(t) + 0.08 * np.cos(2.0 * t)
    h_blk = np.maximum(h_blk, 0.0)
    candidates.append(h_blk)
    
    # 5. Local Refinement using L-BFGS-B in log-space with smooth objective
    # Sort candidates by their current value (best first) to prioritize good starts
    scored_candidates = []
    for h_cand in candidates:
        h_cand = np.asarray(h_cand, dtype=float)
        if len(h_cand) != N:
            continue
        h_cand = np.maximum(h_cand, 1e-6)
        val_cand = compute_value(h_cand)
        scored_candidates.append((val_cand, h_cand))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Limit number of local refinement attempts to stay within time budget
    max_local_attempts = 8
    attempts = 0
    for val_cand, h_cand in scored_candidates:
        if time.time() > deadline - 20.0:
            break
        if attempts >= max_local_attempts:
            break
        attempts += 1
        
        # Log space start
        log_h_start = np.log(h_cand)
        
        try:
            # Use L-BFGS-B with smooth objective
            res = minimize(
                neg_value,
                log_h_start,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'maxfun': 8000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_refined = np.exp(res.x - np.mean(res.x))
            h_refined = np.maximum(h_refined, 0.0)
            val_refined = compute_value(h_refined)
            
            if val_refined > best_val:
                best_val = val_refined
                best_h = h_refined.copy()
                
        except Exception:
            pass
    
    # 6. Final Polish with exact objective on the absolute best
    if time.time() < deadline - 5.0:
        log_h_best = np.log(np.maximum(best_h, 1e-6))
        try:
            # First pass with smooth objective
            res = minimize(
                neg_value,
                log_h_best,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            h_mid = np.exp(res.x - np.mean(res.x))
            h_mid = np.maximum(h_mid, 0.0)
            
            # Second pass with exact objective for final precision
            log_h_mid = np.log(np.maximum(h_mid, 1e-6))
            res2 = minimize(
                neg_value_exact,
                log_h_mid,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            h_final = np.exp(res2.x - np.mean(res2.x))
            h_final = np.maximum(h_final, 0.0)
            val_final = compute_value(h_final)
            
            if val_final > best_val:
                best_val = val_final
                best_h = h_final.copy()
        except Exception:
            pass
            
    # Ensure output validity
    best_h = np.asarray(best_h, dtype=float)
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 1.0)
    
    return list(best_h)

# Test block (optional, remove for submission if not needed, but good for sanity)
if __name__ == "__main__":
    h = construct()
    val = compute_value(h)
    print(f"Value: {val}")
    print(f"Height sample: {h[:5]}")
    print(f"Norm1: {np.sum(np.convolve(h, h)) / (m + 1)}")
    print(f"NormInf: {np.max(np.convolve(h, h))}")
