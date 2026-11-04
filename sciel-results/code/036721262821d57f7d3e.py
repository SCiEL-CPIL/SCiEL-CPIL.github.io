"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.signal import fftconvolve

N = 600
TIME_LIMIT = 110.0

# EVOLVE-BLOCK-START

def _score(heights):
    """Compute the normalized maximum of autoconvolution."""
    h = np.asarray(heights, dtype=float)
    if np.any(h < 0):
        return 1e10
    dx = 0.5 / N
    int_f = np.sum(h) * dx
    if int_f <= 0:
        return 1e10
    
    # Discrete autoconvolution
    c = fftconvolve(h, h, mode='full')
    max_auto = np.max(c) * dx
    
    return max_auto / (int_f * int_f)

def construct():
    start_time = time.time()
    
    # 1. Initial candidates
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/N
    for sigma in [0.05, 0.08, 0.1, 0.15]:
        g = np.exp(-x**2 / (2 * sigma**2))
        s = _score(g)
        if s < best_score:
            best_score = s
            best_h = g.copy()

    # 2. Fourier Optimization
    K = 20
    n_params = 1 + 2 * K
    
    I = np.arange(N)
    basis = np.zeros((n_params, N))
    basis[0, :] = 1.0
    for k in range(1, K + 1):
        angle = 2 * np.pi * k * I / N
        basis[2*k - 1, :] = np.cos(angle)
        basis[2*k, :] = np.sin(angle)
    
    def fourier_to_heights(params):
        return np.dot(params, basis)
    
    def objective_fourier(params):
        h = fourier_to_heights(params)
        h_clipped = np.maximum(0, h)
        penalty = np.sum(np.minimum(0, h)**2) * 100
        return _score(h_clipped) + penalty
    
    bounds = [(0, 1)] + [(-1, 1)] * (2 * K)
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 60:
        return list(best_h)
        
    try:
        result = differential_evolution(
            objective_fourier, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=0.01,
            seed=42,
            polish=False
        )
        
        h_new = fourier_to_heights(result.x)
        h_new = np.maximum(0, h_new)
        s_new = _score(h_new)
        
        if s_new < best_score:
            best_score = s_new
            best_h = h_new.copy()
            
    except Exception as e:
        pass

    # 3. Local Refinement on Heights
    def objective_heights(h_flat):
        h = np.maximum(0, h_flat)
        return _score(h)
        
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 20:
        return list(best_h)
        
    try:
        result = minimize(
            objective_heights,
            best_h,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_h = np.maximum(0, result.x)
            
    except Exception as e:
        pass

    return list(best_h)

# EVOLVE-BLOCK-END
