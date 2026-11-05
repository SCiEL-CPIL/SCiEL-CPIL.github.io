"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute base frequencies for efficiency
_n = np.arange(N)
_TWO_PI = 2 * np.pi

def _value(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / s**2)

def _make_h_from_fourier(params, K):
    """
    Construct h from Fourier coefficients.
    params: [a1, b1, a2, b2, ..., aK, bK]
    h[n] = (1 + sum(a_k cos(...) + b_k sin(...))) / N
    This ensures sum(h) = 1.
    """
    h = np.ones(N)
    for k in range(1, K + 1):
        a = params[2 * (k - 1)]
        b = params[2 * (k - 1) + 1]
        h += a * np.cos(_TWO_PI * k * _n / N) + b * np.sin(_TWO_PI * k * _n / N)
    return h / N

def _smooth_max(x, beta=100.0):
    """Soft max approximation."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)) / len(x)) / beta

def _objective(params, K, beta):
    """
    Objective function for optimization.
    Uses smooth max to ensure differentiability.
    """
    h = _make_h_from_fourier(params, K)
    # sum(h) is exactly 1 by construction, but let's be safe for numerical errors
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    
    conv = np.convolve(h, h)
    # We want to minimize 2 * N * max(conv) / s^2
    # Since s=1, we minimize 2*N*max(conv)
    # Using smooth max for differentiability
    sm = _smooth_max(conv, beta)
    return 2.0 * N * sm / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    # We will optimize over Fourier coefficients
    # K=5 modes -> 10 parameters. 
    # Higher K allows more shape freedom but harder optimization.
    # K=5 is a good balance.
    K = 5
    dim = 2 * K
    
    # Bounds for coefficients to prevent h from becoming too spiky or negative in a way that hurts
    # h[n] should be positive ideally to keep sum stable, but task allows negative.
    # However, large oscillations increase max(conv).
    bounds = [(-1.0, 1.0)] * dim
    
    rng = np.random.default_rng(42)
    
    # Multi-start optimization
    n_starts = 30
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        # Initialize parameters
        if i == 0:
            x0 = np.zeros(dim)
        else:
            # Random start, scaled down to start near the seed solution
            x0 = rng.uniform(-0.5, 0.5, size=dim)
        
        # Use L-BFGS-B with a smooth max surrogate
        # Start with lower beta (smoother) and potentially refine with higher beta
        # For simplicity, we use a fixed moderate beta which is usually sufficient
        # for finding the basin, then we evaluate the true value.
        beta = 50.0
        
        try:
            res = minimize(
                _objective,
                x0,
                args=(K, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.x is not None:
                h_candidate = _make_h_from_fourier(res.x, K)
                val_candidate = _value(h_candidate)
                
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate.copy()
                    
                    # If we found a good one, try to refine with a steeper max (higher beta)
                    # This helps if the smooth max was too flat
                    if time.time() < deadline - 5.0:
                        beta_refine = 200.0
                        res2 = minimize(
                            _objective,
                            res.x,
                            args=(K, beta_refine),
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                        )
                        if res2.x is not None:
                            h_refined = _make_h_from_fourier(res2.x, K)
                            val_refined = _value(h_refined)
                            if val_refined < best_val:
                                best_val = val_refined
                                best_h = h_refined.copy()
                                
        except Exception:
            continue
            
    # Final local search: simple gradient-free perturbation on the best Fourier coeffs?
    # Or just return best_h. The optimization should be sufficient.
    
    return best_h
# EVOLVE-BLOCK-END
