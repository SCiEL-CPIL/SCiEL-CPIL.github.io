import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 600

def compute_autoconv(h):
    """Compute linear autoconvolution of h using FFT."""
    h = np.asarray(h, dtype=np.float64)
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    return conv

def compute_score(h):
    """Compute normalized max autoconvolution score."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return np.inf
    h = np.maximum(h, 0.0)
    sum_h = np.sum(h)
    if sum_h <= 0:
        return np.inf
    
    conv = compute_autoconv(h)
    max_conv = np.max(conv)
    score = max_conv / (sum_h * sum_h)
    return score

def params_to_h(params, M=3):
    """Convert Gaussian mixture parameters to step heights."""
    h = np.zeros(N)
    k = np.arange(N)
    for j in range(M):
        A = params[3*j]
        c = params[3*j+1]
        s = params[3*j+2]
        
        # Ensure width is positive
        s = abs(s) + 1e-6
        # Ensure amplitude is non-negative
        A = max(A, 0.0)
        
        # Gaussian component
        g = A * np.exp(-0.5 * ((k - c) / s) ** 2)
        h += g
        
    # Normalize to sum=1 to make objective scale-invariant and well-conditioned
    sum_h = np.sum(h)
    if sum_h > 0:
        h = h / sum_h
    else:
        h = np.ones(N) / N
        
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    M = 3
    n_params = 3 * M
    
    # Bounds: [A, c, s] for each component
    bounds = [
        (0.1, 10.0), 
        (-10.0, N + 10.0), 
        (1.0, 100.0)
    ] * M

    best_h = np.ones(N)
    best_score = compute_score(best_h)

    def obj_func(params):
        if time.time() > deadline:
            return best_score
        h = params_to_h(params, M)
        return compute_score(h)

    try:
        # Phase 1: Global Search with Differential Evolution
        res_de = differential_evolution(
            obj_func, 
            bounds, 
            maxiter=50, 
            popsize=20, 
            seed=42, 
            polish=False,
            tol=1e-5
        )
        
        if time.time() > deadline - 15:
            return list(best_h)

        # Update best from DE
        h_de = params_to_h(res_de.x, M)
        score_de = compute_score(h_de)
        if score_de < best_score:
            best_h = h_de
            best_score = score_de

        # Phase 2: Local Polish with Powell
        res_local = minimize(
            obj_func, 
            res_de.x, 
            method='Powell', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'xtol': 1e-8}
        )
        
        if res_local.fun < best_score:
            best_h = params_to_h(res_local.x, M)
            best_score = res_local.fun

    except Exception:
        pass

    if time.time() > deadline:
        return list(best_h)

    # Final safety check
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
