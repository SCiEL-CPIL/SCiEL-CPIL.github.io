"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """CMA-ES for BBOB-style Rosenbrock in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_fx = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls
            # Ensure x is a finite 1D array of size n
            if not np.all(np.isfinite(x)) or x.shape != (n,):
                return np.inf
            if calls >= budget:
                return np.inf
            val = func(x)
            calls += 1
            return val

        # Phase 1: Initial Exploration
        # Use a small number of random points to find a good starting point
        n_initial = min(20, budget // 5)
        initial_points = []
        for _ in range(n_initial):
            x = rng.uniform(-4, 4, n)
            f = safe_eval(x)
            if np.isfinite(f):
                initial_points.append((f, x))
                if f < best_fx:
                    best_fx = f
                    best_x = x.copy()
            
            if calls >= budget or time.time() > deadline:
                break

        if not initial_points:
            # Fallback if no valid points
            x = np.zeros(n)
            f = safe_eval(x)
            if np.isfinite(f):
                best_fx = f
                best_x = x.copy()
            return best_x if best_x is not None else np.zeros(n), best_fx

        # Sort initial points and pick top 3 for local optimization
        initial_points.sort(key=lambda t: t[0])
        top_points = [p[1] for p in initial_points[:3]]

        # Phase 2: Local Optimization with Nelder-Mead
        # We use scipy.optimize.minimize with Nelder-Mead method
        # We need to manage the budget carefully.
        # Nelder-Mead can take many iterations. We limit the number of function calls
        # by using a custom callback or just limiting the maxiter if possible, 
        # but Nelder-Mead doesn't have a strict maxfun in older scipy.
        # Instead, we wrap the function to count calls and stop if we exceed budget.
        
        current_calls = calls
        
        def wrapped_func(x):
            nonlocal current_calls
            if current_calls >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            current_calls += 1
            return val

        for start_x in top_points:
            if time.time() > deadline:
                break
            if current_calls >= budget:
                break
            
            try:
                # Use Nelder-Mead
                # Initial simplex size is automatically determined by scipy, 
                # but we can set xatol and fatol to stop early.
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    start_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
                )
                if np.isfinite(res.fun) and res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Ensure we return a valid point
        if best_x is None:
            best_x = np.zeros(n)
            best_fx = safe_eval(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
