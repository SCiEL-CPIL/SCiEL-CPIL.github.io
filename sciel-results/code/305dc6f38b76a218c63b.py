"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s limit

# EVOLVE-BLOCK-START
def compute_value(h):
    """
    Computes the objective value for a given height array h.
    h: numpy array of length N, non-negative.
    """
    g = np.convolve(h, h)
    m = len(g)
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Spacing w
    # Interval [-0.5, 0.5] length 1. m+1 intervals.
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # Vectorized calculation
    y_curr = y[:-1]
    y_next = y[1:]
    term = y_curr**2 + y_curr * y_next + y_next**2
    l2 = (w / 3.0) * np.sum(term)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def objective(h):
    """
    Objective function for minimization (returns negative value).
    Normalizes h to unit L2 norm to handle scale invariance.
    """
    norm = np.linalg.norm(h)
    if norm < 1e-12:
        return 0.0
    h_norm = h / norm
    val = compute_value(h_norm)
    return -val

def construct():
    global START_TIME
    
    # Initial seed
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Generate diverse initial guesses
    x = np.linspace(-0.25, 0.25, N)
    initial_guesses = []
    
    # 1. Uniform (Seed)
    initial_guesses.append(np.ones(N))
    
    # 2. Gaussians with different widths
    for sigma in [0.05, 0.08, 0.10, 0.12, 0.15]:
        h_gauss = np.exp(-x**2 / (2 * sigma**2))
        initial_guesses.append(h_gauss)
        
    # 3. Cosine bump
    h_cos = 0.5 * (1 + np.cos(2 * np.pi * x))
    initial_guesses.append(h_cos)
    
    # 4. Polynomial bump (smooth)
    # (1 - (2x)^4)^2, zero at edges
    h_poly = (1 - (2*x)**4)**2
    h_poly = np.maximum(h_poly, 0)
    initial_guesses.append(h_poly)
    
    # 5. Triangle (Autocorrelation of uniform is triangle, but we want h itself)
    h_tri = 1 - np.abs(2*x)
    initial_guesses.append(h_tri)

    bounds = [(0, None)] * N
    
    for h0 in initial_guesses:
        # Check time
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Ensure non-negative for L-BFGS-B
        h0 = np.maximum(h0, 0)
        
        # If h0 is zero, skip
        if np.sum(h0) == 0:
            continue
            
        try:
            # Run optimization
            # L-BFGS-B is robust for bounded problems
            # We use a moderate number of iterations
            res = minimize(
                objective, 
                h0, 
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            if res.x is not None:
                # Evaluate the result
                h_opt = res.x
                val_opt = -res.fun
                
                # Also check the raw h0 in case optimizer failed or moved to boundary
                val_h0 = -objective(h0)
                
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
                elif val_h0 > best_val:
                    best_val = val_h0
                    best_h = h0
                    
        except Exception:
            pass

    # Final safety: ensure finite and non-negative
    best_h = np.maximum(best_h, 0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return list(best_h)
# EVOLVE-BLOCK-END
