import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
w = 1.0 / 100.0
m = 2 * N - 1  # 99
m_plus_1 = m + 1  # 100

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
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft <<= 1
        
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:m]
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

def compute_value_smooth(h, smoothing=1e-6):
    """
    Compute the objective value with smoothed max for better gradient flow.
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    
    # FFT based convolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft <<= 1
        
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:m]
    g = np.real(g)
    
    # Clip small negative values from numerical noise
    g = np.maximum(g, 0.0)
    
    # Calculate components
    sum_g_sq = np.sum(g**2)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    sum_g = np.sum(g)
    
    # Smooth max using log-sum-exp
    max_g = np.max(g)
    if max_g > 0:
        # Soft max approximation
        smooth_max = max_g + smoothing * np.log(np.sum(np.exp((g - max_g) / smoothing)))
    else:
        smooth_max = max_g
    
    if smooth_max == 0 or sum_g == 0:
        return 0.0
        
    numerator = w * m_plus_1 * (sum_g_sq + 2.0 * sum_g_g_next)
    denominator = 3.0 * sum_g * smooth_max
    
    return numerator / denominator

def neg_value(log_h):
    """
    Objective function for minimization in log-space.
    """
    # Center to prevent overflow/underflow issues in exp
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value(h)

def neg_value_smooth(log_h, smoothing=1e-4):
    """
    Objective function for minimization in log-space with smoothed max.
    """
    # Center to prevent overflow/underflow issues in exp
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value_smooth(h, smoothing)

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

def optimize_parametric(func, bounds, initial_guess=None):
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
        res = differential_evolution(obj, bounds, seed=42, maxiter=200, tol=1e-8, polish=True)
        best_params = res.x
        best_h = func(best_params)
        best_val = compute_value(best_h)
        return best_val, best_h, best_params
    except Exception:
        return 0.0, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s buffer
    
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
    
    # 2. Parametric Global Optimization
    # Define bounds:
    # sigma: 1 to 20 -> log: 0 to 3
    # beta: 1 to 5
    # R: 5 to 30 -> log: 1.6 to 3.4
    # p: 1 to 5
    # w_width: 10 to 50 -> log: 2.3 to 3.9
    
    families = [
        ("gauss", parametric_h_gaussian, [[0.0, 3.0]]),
        ("super_gauss", parametric_h_super_gaussian, [[0.0, 3.0], [1.0, 5.0]]),
        ("poly", parametric_h_polynomial, [[1.5, 3.5], [1.0, 5.0]]),
        ("flat_top", parametric_h_flat_top, [[2.0, 4.0]]),
    ]
    
    for name, func, bounds in families:
        if time.time() > deadline - 30.0:
            break
        val, h, _ = optimize_parametric(func, bounds)
        if h is not None and val > best_val:
            best_val = val
            best_h = h.copy()
        if h is not None:
            candidates.append(h)
            
    # 3. Add some heuristic candidates
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
    
    # 4. Local Refinement using L-BFGS-B in log-space
    # We will refine each candidate.
    # To save time, we can skip candidates that are clearly worse after a quick check,
    # but local opt is cheap, so we do all if time permits.
    
    for h_cand in candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Ensure valid start
        h_cand = np.maximum(h_cand, 1e-6)
        
        # Log space start
        log_h_start = np.log(h_cand)
        
        try:
            # Use L-BFGS-B with smoothed objective first
            res = minimize(
                neg_value_smooth,
                log_h_start,
                args=(1e-3,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_refined = np.exp(res.x - np.mean(res.x))
            h_refined = np.maximum(h_refined, 0.0)
            val_refined = compute_value(h_refined)
            
            if val_refined > best_val:
                best_val = val_refined
                best_h = h_refined.copy()
                
        except Exception:
            pass
    
    # 5. Final Polish with more iterations on the absolute best
    if time.time() < deadline - 5.0:
        log_h_best = np.log(np.maximum(best_h, 1e-6))
        try:
            # First pass with smoothed objective
            res = minimize(
                neg_value_smooth,
                log_h_best,
                args=(1e-5,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_intermediate = np.exp(res.x - np.mean(res.x))
            h_intermediate = np.maximum(h_intermediate, 0.0)
            
            # Second pass with exact objective
            log_h_intermediate = np.log(np.maximum(h_intermediate, 1e-6))
            res = minimize(
                neg_value,
                log_h_intermediate,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            h_final = np.exp(res.x - np.mean(res.x))
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
