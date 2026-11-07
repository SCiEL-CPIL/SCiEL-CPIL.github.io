"""Hybrid optimizer for BBOB f19: stratified global search + accelerated local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using a deterministic multi-start strategy:
    1. Stratified Latin Hypercube Sampling for global coverage
    2. Accelerated BFGS/Nelder-Mead local search from each basin
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        calls_made = 0
        best_x = None
        best_f = np.inf
        all_x = []
        all_f = []

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                x = np.asarray(x, dtype=np.float64)
                if x.shape != (dim,):
                    x = x.reshape(dim)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                if not np.isfinite(f):
                    f = 1e10
                calls_made += 1
                all_x.append(x.copy())
                all_f.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None

        # --- Phase 1: Stratified Latin Hypercube Sampling for global coverage ---
        # Use ~40% of budget for global exploration
        global_budget = int(budget * 0.4)
        n_lhs = min(global_budget, 100)  # Cap at 100 for efficiency
        
        # Generate LHS points
        rng = np.random.default_rng(42)
        lhs_points = []
        
        # Standard LHS: divide each dimension into n_lhs equal intervals
        for i in range(n_lhs):
            point = np.zeros(dim)
            for d in range(dim):
                # Randomly permute the intervals for this dimension
                if d == 0:
                    perm = rng.permutation(n_lhs)
                else:
                    perm = rng.permutation(n_lhs)
                # Pick the interval corresponding to row i
                interval = perm[i]
                # Uniform within that interval
                lower = -5.0 + (10.0 * interval) / n_lhs
                upper = -5.0 + (10.0 * (interval + 1)) / n_lhs
                point[d] = rng.uniform(lower, upper)
            lhs_points.append(point)
        
        # Add center point
        lhs_points.append(np.zeros(dim))
        
        # Evaluate all LHS points
        lhs_results = []
        for x in lhs_points:
            if calls_made >= budget or time.time() > deadline - 10.0:
                break
            f = safe_call(x)
            if f is not None:
                lhs_results.append((x, f))
        
        if not lhs_results:
            # Fallback if all calls failed
            return np.zeros(dim), 1e10

        # --- Phase 2: Local refinement from top basins ---
        # Sort by fitness and pick top candidates for local search
        lhs_results.sort(key=lambda x: x[1])
        
        # Select top K basins for local refinement
        # Use remaining budget for local search
        local_budget = budget - calls_made
        n_basins = min(10, len(lhs_results))  # Up to 10 basins
        
        # Allocate local budget among basins
        budget_per_basin = max(5, local_budget // n_basins)
        
        for i in range(n_basins):
            if calls_made >= budget - 5 or time.time() > deadline - 5.0:
                break
            
            x_start, f_start = lhs_results[i]
            
            # Try multiple local optimization methods
            # 1. Nelder-Mead (robust, derivative-free)
            try:
                nm_calls = 0
                def nm_func(x):
                    nonlocal nm_calls
                    if nm_calls >= budget_per_basin or calls_made >= budget:
                        return 1e10
                    f = safe_call(x)
                    if f is None:
                        return 1e10
                    nm_calls += 1
                    return f
                
                res = minimize(nm_func, x_start, method='Nelder-Mead',
                              options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_f and np.isfinite(res.fun):
                    best_x = np.clip(res.x, -5.0, 5.0)
                    best_f = res.fun
            except Exception:
                pass
            
            # 2. Powell method (good for smooth functions)
            if calls_made < budget - 5 and time.time() < deadline - 5.0:
                try:
                    pw_calls = 0
                    def pw_func(x):
                        nonlocal pw_calls
                        if pw_calls >= budget_per_basin // 2 or calls_made >= budget:
                            return 1e10
                        f = safe_call(x)
                        if f is None:
                            return 1e10
                        pw_calls += 1
                        return f
                    
                    res = minimize(pw_func, x_start, method='Powell',
                                  options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-8})
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_x = np.clip(res.x, -5.0, 5.0)
                        best_f = res.fun
                except Exception:
                    pass
        
        # --- Phase 3: Final polish with remaining budget ---
        if best_x is not None and calls_made < budget and time.time() < deadline - 2.0:
            remaining = budget - calls_made
            if remaining > 10:
                try:
                    final_calls = 0
                    def final_func(x):
                        nonlocal final_calls
                        if final_calls >= remaining or calls_made >= budget:
                            return 1e10
                        f = safe_call(x)
                        if f is None:
                            return 1e10
                        final_calls += 1
                        return f
                    
                    # Final high-precision Nelder-Mead
                    res = minimize(final_func, best_x, method='Nelder-Mead',
                                  options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_x = np.clip(res.x, -5.0, 5.0)
                        best_f = res.fun
                except Exception:
                    pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)
            if best_f is None:
                best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END
