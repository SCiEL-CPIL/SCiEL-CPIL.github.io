"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Memetic optimizer: Random sampling + Local L-BFGS-B refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # 300s limit, 10s safety margin
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        total_budget = self.budget
        bounds = [(-5.0, 5.0) for _ in range(dim)]

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Diversified Initialization
        # Sample enough points to likely hit the global basin of f6
        n_initial = min(50, total_budget // 2)
        if n_initial < 10:
            n_initial = 10
            
        initial_points = []
        # Include center
        initial_points.append(np.zeros(dim))
        for _ in range(n_initial - 1):
            x0 = rng.uniform(-5, 5, dim)
            initial_points.append(x0)

        init_results = []
        for x0 in initial_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            fx = safe_func(x0)
            if np.isfinite(fx):
                init_results.append((fx, x0))

        if not init_results:
            # Fallback if all failed or budget exhausted
            x0 = np.zeros(dim)
            fx = safe_func(x0)
            return best_x, best_f

        # Sort initial results
        init_results.sort(key=lambda t: t[0])
        
        # Phase 2: Local Optimization from the best starts
        # We take the top K starts and run L-BFGS-B on them
        # L-BFGS-B is efficient for smooth functions like f6
        k_local = 5
        for i in range(k_local):
            if calls_made >= total_budget or time.time() > deadline:
                break
            if i >= len(init_results):
                break
                
            fx_init, x_init = init_results[i]
            
            # Run local optimization
            # maxiter is set high, but the safe_func will stop it if budget runs out
            try:
                res = minimize(safe_func, x_init, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 1000, 'maxfun': total_budget - calls_made})
                # The minimize function might not respect the global deadline perfectly if 
                # it's in a tight loop, but safe_func returns inf which should stop it or 
                # the time check in safe_func prevents further calls.
            except Exception:
                pass
            
            # Check if we should continue
            if time.time() > deadline:
                break

        # Phase 3: If we have budget left, do a few more random restarts + local search
        # This helps if the first batch of starts missed the global basin
        while calls_made < total_budget and time.time() < deadline:
            # Generate a new random point
            x0 = rng.uniform(-5, 5, dim)
            fx0 = safe_func(x0)
            
            if np.isinf(fx0):
                break
                
            # If this point is significantly better than current best, or just for diversity
            # Run a short local optimization
            if calls_made + 10 < total_budget: # Ensure we have some budget for local search
                try:
                    res = minimize(safe_func, x0, method='L-BFGS-B', bounds=bounds, 
                                   options={'maxiter': 50, 'maxfun': 20})
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
