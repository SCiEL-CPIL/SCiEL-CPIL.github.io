"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead based optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        # The evaluator kills at 330s, but the contract says 300s limit for the function.
        # We use a tight deadline to ensure we return before any hard kill.
        deadline = start_time + 295 
        d = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls
            if calls >= budget or not time_ok():
                return 1e10
            calls += 1
            try:
                val = func(x)
                if np.isnan(val) or np.isinf(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        # Phase 1: Global Exploration
        # Discus (f11) has a narrow valley. Random sampling in 5D is inefficient.
        # We use a combination of center, corners, and stratified random points.
        n_starts = 15
        rng = np.random.default_rng(42)
        
        start_points = []
        # Center
        start_points.append(np.zeros(d))
        
        # Corners (subset)
        corners = []
        for i in range(4):
            pt = np.zeros(d)
            pt[i] = 5.0
            start_points.append(pt)
            pt[i] = -5.0
            start_points.append(pt)
            
        # Fill remaining with random
        while len(start_points) < n_starts:
            start_points.append(rng.uniform(-5, 5, d))

        # Evaluate all start points
        start_fits = []
        best_x = start_points[0].copy()
        best_f = wrapped_func(best_x)
        start_fits.append(best_f)
        
        for x in start_points[1:]:
            if not time_ok() or calls >= budget:
                break
            fx = wrapped_func(x)
            start_fits.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort start points by fitness
        sorted_indices = np.argsort(start_fits)
        sorted_starts = [start_points[i] for i in sorted_indices]
        sorted_fits = [start_fits[i] for i in sorted_indices]

        # Phase 2: Local Refinement with Nelder-Mead
        # Nelder-Mead is derivative-free and robust.
        # We run it on the top 5 starting points.
        # We need to be careful with the call budget.
        # NM in 5D typically takes ~50-100 calls to converge well.
        # 5 starts * 100 calls = 500 calls. This fits the budget.
        
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_f

        n_refine = 5
        # Allocate budget per start. If we have 500 calls left and 5 starts, ~100 each.
        # But we must stop if budget is exhausted.
        
        for i in range(n_refine):
            if not time_ok() or calls >= budget:
                break
            
            x0 = sorted_starts[i]
            f0 = sorted_fits[i]
            
            # Skip if this start is significantly worse than the current best
            # to avoid wasting budget on poor local minima
            # Discus values can vary, but a large gap suggests we are far from the valley.
            if f0 > best_f + 100: 
                continue

            # Run Nelder-Mead
            # Initial simplex size: 0.1 is a reasonable default for [-5,5]
            # If the valley is very narrow, we might need smaller, but 0.1 is safe.
            # maxiter: We limit iterations to ensure we don't exceed budget.
            # 200 iterations is a safe cap.
            
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

            if not time_ok() or calls >= budget:
                break

        # Phase 3: Polish with a very tight Nelder-Mead from the best found
        if not time_ok() or calls >= budget:
            return best_x, best_f

        # If we have budget left, do one more precise run from the best point
        if calls < budget - 10:
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
