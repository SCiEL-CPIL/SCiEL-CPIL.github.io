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

def objective_sym(v):
    """
    Objective function for symmetric h.
    v: numpy array of length N//2 (25).
    Returns negative value.
    """
    # Expand to full h
    h = np.concatenate([v, v[::-1]])
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    # Normalize to avoid scale issues, though the ratio is scale invariant
    # Wait, is it scale invariant?
    # l2 is quadratic in h (since g is quadratic in h, y is quadratic, l2 is quartic? No.)
    # g ~ h^2. y ~ h^2. l2 ~ h^4.
    # norm_1 ~ h^2. norm_inf ~ h^2.
    # Ratio ~ h^4 / (h^2 * h^2) = constant.
    # So it IS scale invariant.
    # However, numerically, scaling can matter.
    # Let's normalize to unit L2 norm of h.
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
    
    # We will optimize the first half of h, assuming symmetry.
    # h[i] = h[N-1-i]
    # v = h[0:N//2]
    # h = [v, v[::-1]]
    
    x = np.linspace(-0.25, 0.25, N)
    half_x = x[:N//2]
    
    initial_guesses_v = []
    
    # 1. Uniform
    initial_guesses_v.append(np.ones(N//2))
    
    # 2. Gaussians
    for sigma in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        h_gauss = np.exp(-x**2 / (2 * sigma**2))
        h_gauss = np.maximum(h_gauss, 0)
        v = h_gauss[:N//2]
        initial_guesses_v.append(v)
        
    # 3. Cosine bump
    h_cos = 0.5 * (1 + np.cos(2 * np.pi * x))
    h_cos = np.maximum(h_cos, 0)
    v = h_cos[:N//2]
    initial_guesses_v.append(v)
    
    # 4. Polynomial bump
    h_poly = (1 - (2*x)**4)**2
    h_poly = np.maximum(h_poly, 0)
    v = h_poly[:N//2]
    initial_guesses_v.append(v)
    
    # 5. Triangle
    h_tri = 1 - np.abs(2*x)
    h_tri = np.maximum(h_tri, 0)
    v = h_tri[:N//2]
    initial_guesses_v.append(v)
    
    # 6. Random symmetric starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        v_rand = rng.random(N//2)
        initial_guesses_v.append(v_rand)

    bounds = [(0, None)] * (N//2)
    
    for v0 in initial_guesses_v:
        # Check time
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Ensure non-negative
        v0 = np.maximum(v0, 0)
        
        # If v0 is zero, skip
        if np.sum(v0) == 0:
            continue
            
        try:
            # Run optimization
            # Using Powell which is derivative free and robust
            res = minimize(
                objective_sym, 
                v0, 
                method='Powell',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-10,
                    'xtol': 1e-10
                }
            )
            
            if res.x is not None:
                # Reconstruct h
                h_opt = np.concatenate([res.x, res.x[::-1]])
                h_opt = np.maximum(h_opt, 0)
                val_opt = compute_value(h_opt)
                
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
                    
        except Exception:
            pass

    # Final safety: ensure finite and non-negative
    best_h = np.maximum(best_h, 0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return list(best_h)
# EVOLVE-BLOCK-END
