import numpy as np
from scipy.optimize import minimize
import time

def _union_len_1d(positions, c, beta=30.0):
    """Smooth 1D union length using soft-max of coverage indicators.
    
    positions: shape (N,), base positions
    c: height parameter
    Returns: smooth approximation of union length of N intervals of width 1/N
    """
    N = len(positions)
    frac = 1.0 / N
    x0s = positions + c * np.arange(N) * frac
    x1s = x0s + frac
    
    # Sort by left endpoint
    order = np.argsort(x0s)
    x0s_s = x0s[order]
    x1s_s = x1s[order]
    
    # Union length = sum of (x1_i - x0_i) - overlaps
    # Overlap between consecutive intervals in sorted order: max(0, min(x1_i, x1_{i+1}) - max(x0_i, x0_{i+1}))
    # For sorted intervals: overlap_i = max(0, x1s_s[i] - x0s_s[i+1]) for i < N-1
    # But this only works if intervals are "chain-like". More generally:
    # Union length = x1s_s[-1] - x0s_s[0] - sum of gaps
    # Gaps: gap_i = max(0, x0s_s[i+1] - x1s_s[i])
    
    gaps = np.maximum(0, x0s_s[1:] - x1s_s[:-1])
    # Smooth max(0, z) = log(1 + exp(beta*z)) / beta
    smooth_gaps = np.log1p(np.exp(beta * gaps)) / beta
    
    total_span = x1s_s[-1] - x0s_s[0]
    union_len = total_span - np.sum(smooth_gaps)
    
    return union_len

def _union_area_at_c(x, y, c, beta=30.0):
    """Smooth union area of 8x8 rectangles at height c.
    Area = union_len_x * union_len_y (since rectangles are axis-aligned and independent)
    """
    len_x = _union_len_1d(x, c, beta)
    len_y = _union_len_1d(y, c, beta)
    return len_x * len_y

def _cost_func(params, beta=30.0):
    x = params[:8]
    y = params[8:]
    c_samples = np.linspace(0.05, 0.95, 15)
    total_area = 0.0
    for c in c_samples:
        total_area += _union_area_at_c(x, y, c, beta)
    return total_area / len(c_samples)

def _cost_grad(params, beta=30.0):
    """Gradient of cost function via finite differences (central)."""
    x = params[:8]
    y = params[8:]
    c_samples = np.linspace(0.05, 0.95, 10)
    eps = 1e-5
    grad = np.zeros_like(params)
    
    for idx in range(16):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[idx] += eps
        params_minus[idx] -= eps
        grad[idx] = (_cost_func(params_plus, beta) - _cost_func(params_minus, beta)) / (2 * eps)
    
    return grad

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    N = 8
    deadline = 590  # safety margin before 600s limit
    
    best_params = None
    best_cost = np.inf
    
    def check_time():
        return time.time() > deadline
    
    # Phase 1: Annealing with smooth objective
    # Start with a reasonable configuration and gradually cool
    n_restarts = 8
    for restart in range(n_restarts):
        if check_time():
            break
            
        # Different initializations
        rng = np.random.default_rng(1000 + restart * 137)
        if restart == 0:
            # Grid-like
            x0 = np.linspace(0, 1 - 1/N, N)
            y0 = np.linspace(0, 1 - 1/N, N)
        elif restart == 1:
            # Slightly perturbed grid
            x0 = np.linspace(0, 1 - 1/N, N) + rng.uniform(-0.05, 0.05, N)
            y0 = np.linspace(0, 1 - 1/N, N) + rng.uniform(-0.05, 0.05, N)
        elif restart == 2:
            # Concentrated
            x0 = rng.uniform(0.2, 0.8, N)
            y0 = rng.uniform(0.2, 0.8, N)
        elif restart == 3:
            # Spread
            x0 = rng.uniform(-0.1, 1.1, N)
            y0 = rng.uniform(-0.1, 1.1, N)
        elif restart == 4:
            # Anti-grid (shifted)
            x0 = np.linspace(0.05, 1 - 1/N - 0.05, N)
            y0 = np.linspace(0.1, 1 - 1/N - 0.1, N)
        elif restart == 5:
            # Random with clustering
            x0 = np.sort(rng.uniform(0, 1, N))
            y0 = np.sort(rng.uniform(0, 1, N))
        elif restart == 6:
            # Alternating pattern
            x0 = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
            y0 = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5])
        else:
            x0 = rng.uniform(0, 1, N)
            y0 = rng.uniform(0, 1, N)
        
        params0 = np.concatenate([x0, y0])
        cost0 = _cost_func(params0, beta=20.0)
        
        if cost0 < best_cost:
            best_cost = cost0
            best_params = params0.copy()
        
        # Annealing: start with low beta (smoother) and increase
        beta_schedule = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
        current_params = params0.copy()
        
        for beta in beta_schedule:
            if check_time():
                break
                
            # Optimize with current beta
            try:
                res = minimize(_cost_func, current_params, args=(beta,),
                              method='L-BFGS-B', 
                              options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-7})
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_params = res.x.copy()
                current_params = res.x
            except Exception:
                pass
        
        # Powell refinement with high beta
        if not check_time():
            try:
                res = minimize(_cost_func, current_params, args=(40.0,),
                              method='Powell',
                              options={'maxiter': 500, 'xtol': 1e-7, 'ftol': 1e-7})
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 2: Final polish with very high beta (nearly exact)
    if best_params is not None and not check_time():
        try:
            res = minimize(_cost_func, best_params, args=(50.0,),
                          method='Powell',
                          options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-8})
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Local search around best solution
    if best_params is not None and not check_time():
        rng = np.random.default_rng(42)
        for trial in range(10):
            if check_time():
                break
            perturbation = rng.normal(0, 0.02, 16)
            candidate = best_params + perturbation
            cand_cost = _cost_func(candidate, beta=50.0)
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_params = candidate.copy()
                # Refine
                try:
                    res = minimize(_cost_func, best_params, args=(50.0,),
                                  method='Powell',
                                  options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8})
                    if res.fun < best_cost:
                        best_cost = res.fun
                        best_params = res.x.copy()
                except Exception:
                    pass
    
    if best_params is None:
        x0 = np.linspace(0, 1 - 1/N, N)
        y0 = np.linspace(0, 1 - 1/N, N)
        best_params = np.concatenate([x0, y0])
    
    x_opt = best_params[:8]
    y_opt = best_params[8:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions
