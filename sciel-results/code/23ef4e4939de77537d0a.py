"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time


class Optimizer:
    """Multi-restart L-BFGS-B with line-search polishing for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit

        call_count = 0
        max_calls = self.budget

        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                return 1e10
            call_count += 1
            x = np.clip(x, -5, 5)
            val = func(x)
            if not np.isfinite(val):
                val = 1e10
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def time_ok():
            return time.time() < deadline - 3

        rng = np.random.default_rng(42)

        # Phase 1: Coarse grid + random restarts with L-BFGS-B
        # Generate a diverse set of initial points
        initial_points = []
        
        # Origin
        initial_points.append(np.zeros(self.dim))
        
        # Grid points: use a structured grid
        # For 5D, use points at -3, 0, 3 in each dimension but that's 3^5=243 which is too many
        # Instead, use a smaller structured set
        grid_vals = [-3.0, -1.5, 0.0, 1.5, 3.0]
        # Use a subset: all combinations but limited
        # Actually, let's use random points plus some structured ones
        
        # Add random points
        n_random = 15
        for _ in range(n_random):
            initial_points.append(rng.uniform(-4.5, 4.5, self.dim))
        
        # Add some axis-aligned points
        for d in range(self.dim):
            for val in [-3.0, 3.0]:
                pt = np.zeros(self.dim)
                pt[d] = val
                initial_points.append(pt)

        # Evaluate all initial points first to rank them
        # But we need to be careful about budget
        # Let's just run L-BFGS-B from each point, but limit the number of restarts
        # based on remaining budget
        
        n_restarts = min(len(initial_points), 20)
        
        # Sort initial points by some heuristic? Or just use them in order
        # Let's evaluate the origin first, then others
        
        for i in range(n_restarts):
            if not time_ok() or call_count >= max_calls:
                break
            
            x0 = initial_points[i].copy()
            
            # Check if we've already evaluated a point close to this
            # For now, just proceed
            
            try:
                remaining = max_calls - call_count
                if remaining < 10:
                    break
                    
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 50,
                        'maxfun': min(remaining, 80),
                        'gtol': 1e-6,
                        'ftol': 1e-8
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                continue

        # Phase 2: Polish the best solution with tighter tolerances
        if best_x is not None and time_ok() and call_count < max_calls:
            remaining = max_calls - call_count
            if remaining >= 10:
                try:
                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': 30,
                            'maxfun': min(remaining, 50),
                            'gtol': 1e-8,
                            'ftol': 1e-10
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        # Phase 3: Try a few more random restarts if budget remains
        if time_ok() and call_count < max_calls:
            for _ in range(5):
                if not time_ok() or call_count >= max_calls:
                    break
                
                x0 = rng.uniform(-4.0, 4.0, self.dim)
                remaining = max_calls - call_count
                if remaining < 10:
                    break
                    
                try:
                    result = minimize(
                        wrapped_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': 30,
                            'maxfun': min(remaining, 40),
                            'gtol': 1e-6,
                            'ftol': 1e-8
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    continue

        # Final check
        if best_x is None:
            best_x = np.zeros(self.dim)
            if call_count < max_calls:
                best_f = wrapped_func(best_x)
            else:
                best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END
