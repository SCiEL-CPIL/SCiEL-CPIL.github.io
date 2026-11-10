import numpy as np
import time
from scipy.optimize import minimize

def calculate_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] using vectorized n^3 sums.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w = weights / np.sum(weights)
    
    # Generate all sums s_i + s_j + s_l
    # This creates an array of size n^3
    s_1 = support[:, None, None]
    s_2 = support[None, :, None]
    s_3 = support[None, None, :]
    sums_123 = (s_1 + s_2 + s_3).flatten()
    
    w_1 = w[:, None, None]
    w_2 = w[None, :, None]
    w_3 = w[None, None, :]
    weights_123 = (w_1 * w_2 * w_3).flatten()
    
    # Sort to allow cumulative sum lookup
    order = np.argsort(sums_123)
    sorted_sums = sums_123[order]
    sorted_weights = weights_123[order]
    cum_weights = np.cumsum(sorted_weights)
    
    P_total = 0.0
    # For each k, threshold is 2 * support[k]
    # We need sum of weights_123 where sums_123 < 2 * support[k]
    for k in range(n):
        thresh = 2.0 * support[k]
        # Find index where sorted_sums >= thresh
        # We want strictly less than, so we take sum of cum_weights up to that index (exclusive)
        idx = np.searchsorted(sorted_sums, thresh, side='left')
        if idx > 0:
            prob_k = cum_weights[idx - 1]
        else:
            prob_k = 0.0
        P_total += w[k] * prob_k
        
    return P_total

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = 580.0  # Safety margin
    
    best_P = -1.0
    best_support = None
    best_weights = None
    
    def update_best(s, w):
        nonlocal best_P, best_support, best_weights
        if s is None or w is None:
            return
        try:
            s = np.asarray(s, dtype=np.float64).flatten()
            w = np.asarray(w, dtype=np.float64).flatten()
            if len(s) != len(w) or len(s) == 0:
                return
            if np.any(np.isnan(s)) or np.any(np.isinf(s)):
                return
            if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                return
            if np.any(s < -1e-9) or np.any(w < -1e-9):
                return
            if np.sum(w) <= 1e-9:
                return
            
            p_val = calculate_P(s, w)
            if p_val > best_P:
                best_P = p_val
                best_support = s.copy()
                best_weights = w.copy()
        except Exception:
            pass

    # Helper to run optimization for a given n and initial guess
    def optimize_candidate(n, s_init, w_init, max_iter=2000):
        def objective(params):
            s = params[:n]
            w = params[n:]
            
            # Clip to valid ranges
            s = np.clip(s, 0, 1e6)
            w = np.clip(w, 0, 1)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 0.0
            w = w / w_sum
            
            return -calculate_P(s, w)
        
        bounds = [(0, 1000)] * n + [(0, 1)] * n
        params_init = np.concatenate([s_init, w_init])
        
        try:
            # Nelder-Mead is robust for this non-smooth objective
            res = minimize(objective, params_init, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if res.fun < -best_P:
                update_best(res.x[:n], res.x[n:])
            return res.x
        except Exception:
            return None

    # Phase 1: Small support sizes with random restarts
    # Small n allows for precise placement of mass to satisfy the inequality.
    rng = np.random.default_rng(42)
    
    for n in [2, 3, 4, 5]:
        # Number of restarts depends on remaining time
        # Estimate time per restart: calculate_P is O(n^3 log n^3). For n=5, ~1000 ops.
        # Minimize takes ~100-1000 evals. So ~1e6 ops. Very fast.
        # We can afford many restarts.
        
        # Initial guesses:
        # 1. Uniform random
        # 2. Log-spaced
        # 3. Based on previous best if available
        
        restarts = 20
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            # Generate random initial support
            # Mix of uniform and log-uniform to explore scales
            if r % 2 == 0:
                s_init = rng.uniform(0, 100, n)
            else:
                s_init = np.exp(rng.uniform(0, 5, n)) # 1 to ~148
            
            # Sort support to help optimizer? Not strictly necessary but good practice
            s_init = np.sort(s_init)
            
            # Generate random weights (Dirichlet)
            w_init = rng.dirichlet(np.ones(n))
            
            optimize_candidate(n, s_init, w_init, max_iter=1000)
            
            # If we found a very good solution, try to refine it with more iterations
            if best_P > 0.5:
                 # Optional: additional polish
                 pass

    # Phase 2: Fixed grid weight optimization
    # This explores the case where many points are used with optimized weights.
    for n_grid in [16, 32]:
        if time.time() > deadline:
            break
            
        s_init = np.linspace(0, 100, n_grid)
        w_init = np.ones(n_grid) / n_grid
        
        def objective_grid(w_flat):
            w = np.clip(w_flat, 0, 1)
            s_sum = np.sum(w)
            if s_sum < 1e-9:
                return 0.0
            w = w / s_sum
            return -calculate_P(s_init, w)
        
        bounds = [(0, 1)] * n_grid
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        try:
            # SLSQP might struggle with non-smoothness, but let's try
            # Or use Nelder-Mead on weights only? Nelder-Mead in high dimensions is slow.
            # Let's try SLSQP first.
            res = minimize(objective_grid, w_init, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-6})
            if res.success:
                update_best(s_init, res.x)
        except Exception:
            pass

    # Phase 3: Final refinement of the best candidate found
    if best_support is not None:
        n_best = len(best_support)
        # Run a few more tight optimizations on the best solution found
        for _ in range(3):
            if time.time() > deadline:
                break
            # Small perturbation
            s_pert = best_support + rng.normal(0, 0.1, n_best)
            s_pert = np.clip(s_pert, 0, 1000)
            w_pert = best_weights + rng.normal(0, 0.01, n_best)
            w_pert = np.clip(w_pert, 0, 1)
            w_pert = w_pert / np.sum(w_pert)
            
            optimize_candidate(n_best, s_pert, w_pert, max_iter=3000)

    if best_support is None:
        # Fallback
        best_support = np.arange(32, dtype=np.float64)
        best_weights = np.ones(32, dtype=np.float64) / 32
        
    return best_support, best_weights
