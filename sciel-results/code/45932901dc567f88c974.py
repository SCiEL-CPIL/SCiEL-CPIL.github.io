import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_value(h):
    """
    Computes the value metric for a given non-negative height vector h.
    """
    h = np.asarray(h, dtype=float)
    
    # Ensure non-negative
    h = np.maximum(h, 0.0)
    
    # Convolve h with itself
    g = np.convolve(h, h, mode='full')
    m = len(g)
    
    # Construct y: [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Spacing w
    w = 1.0 / (m + 1)
    
    # Calculate l2
    term1 = y[:m+1]**2
    term2 = y[:m+1] * y[1:m+2]
    term3 = y[1:m+2]**2
    
    l2 = np.sum((w / 3.0) * (term1 + term2 + term3))
    
    # Calculate norms
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def _objective(x):
    """
    Objective function for optimization.
    x are log-heights.
    """
    # Convert to heights
    h = np.exp(x - np.max(x)) # Scale to prevent overflow, relative shape preserved
    
    # Normalize h to unit L2 norm to remove scale invariance
    norm_h = np.linalg.norm(h)
    if norm_h == 0:
        return 0.0
    h = h / norm_h
    
    val = _compute_value(h)
    return -val

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 115.0 # 120s limit, leave 5s margin
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    center = N / 2.0
    indices = np.arange(N)
    
    guesses = []
    
    # Gaussian guesses
    for sigma in [1.5, 2.5, 4.0, 6.0, 8.0]:
        h_gauss = np.exp(-((indices - center)**2) / (2 * sigma**2))
        h_gauss = h_gauss / np.linalg.norm(h_gauss)
        x0 = np.log(h_gauss + 1e-10)
        guesses.append(x0)
        
    # Raised Cosine guesses
    for width in [8, 16, 24, 40]:
        h_cos = np.zeros(N)
        mask = np.abs(indices - center) <= width/2
        h_cos[mask] = 0.5 * (1 + np.cos(np.pi * (indices[mask] - center) / (width/2)))
        h_cos = h_cos / np.linalg.norm(h_cos)
        x0 = np.log(h_cos + 1e-10)
        guesses.append(x0)
        
    # Uniform guess
    h_unif = np.ones(N)
    h_unif = h_unif / np.linalg.norm(h_unif)
    x0 = np.log(h_unif + 1e-10)
    guesses.append(x0)

    for x0 in guesses:
        if time.time() - start_time > timeout:
            break
            
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            # Check time again
            if time.time() - start_time > timeout:
                break
                
            if res.success:
                h_opt = np.exp(res.x - np.max(res.x))
                norm_h = np.linalg.norm(h_opt)
                if norm_h > 0:
                    h_opt = h_opt / norm_h
                    val_opt = _compute_value(h_opt)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_h = h_opt
        except Exception:
            continue

    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
