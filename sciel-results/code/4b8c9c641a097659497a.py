import numpy as np
from scipy.optimize import differential_evolution

N = 50

def compute_value(h):
    h = np.asarray(h, dtype=float)
    # Ensure non-negative and finite
    h = np.maximum(h, 0.0)
    if np.max(h) < 1e-15:
        return 0.0
    
    # Normalize to max 1 to keep numbers stable, though value is scale invariant
    h_norm = h / np.max(h)
    
    g = np.convolve(h_norm, h_norm)
    m = len(g)
    y = np.concatenate([[0.0], g, [0.0]])
    w = 1.0 / (m + 1)
    
    # Calculate L2 term
    # l2 = sum (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
    y_sq = y ** 2
    y_cross = y[:-1] * y[1:]
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + y_cross + y_sq[1:])
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_1 * norm_inf < 1e-15:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def objective(h):
    return -compute_value(h)

def construct():
    # Differential Evolution is a global optimizer, distinct from Nelder-Mead
    # Bounds: [0, 1] since scale is invariant
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=15,
            tol=0.01,
            polish=True,
            seed=42
        )
        best_h = result.x
    except Exception:
        best_h = np.ones(N)
        
    # Ensure the result is valid and non-negative
    best_h = np.maximum(best_h, 0.0)
    
    # Fallback if DE fails or returns zero
    if np.max(best_h) < 1e-15:
        best_h = np.ones(N)
        
    return list(best_h)
