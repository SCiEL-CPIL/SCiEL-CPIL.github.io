"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        # The evaluator kills at 330s, but the contract says 300s limit for the function.
        # We use a tight deadline to ensure we return before any hard kill.
        deadline = start_time + 290
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

        # Phase 1: Generate diverse starting points
        # Discus (f11) has a narrow valley. L-BFGS-B is very fast if started reasonably close.
        # We use a mix of structured and random points to maximize the chance of hitting the valley.
        
        rng = np.random.default_rng(123)
        
        start_points = []
        
        # 1. Center
        start_points.append(np.zeros(d))
        
        # 2. Axes endpoints (corners of the box along axes)
        for i in range(d):
            pt_pos = np.zeros(d)
            pt_pos[i] = 5.0
            start_points.append(pt_pos)
            
            pt_neg = np.zeros(d)
            pt_neg[i] = -5.0
            start_points.append(pt_neg)
            
        # 3. Random points
        # We have 1 + 2*5 = 11 points so far.
        # Let's add a few more random ones to cover the space better.
        n_random = 5
        for _ in range(n_random):
            start_points.append(rng.uniform(-5, 5, d))
            
        # Total start points: 16
        # Budget is 500.
        # L-BFGS-B in 5D typically converges in < 50-100 iterations.
        # Each iteration might use 1-2 function evaluations (for gradient approximation or line search).
        # 16 starts * ~30 calls = 480 calls. This is tight but feasible.
        # We will prioritize starts that show initial promise.

        # Evaluate all start points first to sort them
        start_fits = []
        for x in start_points:
            if not time_ok() or calls >= budget:
                break
            fx = wrapped_func(x)
            start_fits.append((x, fx))
            
        if not start_fits:
             return np.zeros(d), 1e10

        # Sort by fitness
        start_fits.sort(key=lambda item: item[1])
        
        best_x = start_fits[0][0].copy()
        best_f = start_fits[0][1]
        
        # Phase 2: Refine the best starting points with L-BFGS-B
        # We iterate through the sorted starts. 
        # We allocate a portion of the remaining budget to each.
        # If a start converges quickly, we move to the next.
        # If it uses up its allocation, we move to the next.
        
        remaining_budget = budget - calls
        n_refine = min(len(start_fits), 8) # Refine top 8
        
        # Estimate calls per start. 
        # L-BFGS-B maxiter is usually sufficient. 
        # We'll give each start a max of ~ (remaining_budget / n_refine) calls.
        # But we also need to stop if we find a very good solution.
        
        calls_per_start = max(10, int(remaining_budget / n_refine))
        
        for i in range(n_refine):
            if not time_ok() or calls >= budget:
                break
                
            x0, f0 = start_fits[i]
            
            # If this start is significantly worse than the current best, skip it?
            # L-BFGS-B might still find a better local min, but for Discus, the valley is narrow.
            # If f0 is much worse than best_f, it's likely in a different basin or far away.
            # However, Discus is unimodal-ish but with a narrow valley. 
            # Let's just try the top ones.
            
            # Run L-BFGS-B
            # bounds=[(-5, 5)] * d
            # maxiter: limit to ensure we don't hang
            # ftol, gtol: standard
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', 
                               bounds=[(-5.0, 5.0)] * d,
                               options={'maxiter': calls_per_start, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 20})
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

            if not time_ok() or calls >= budget:
                break

        # Phase 3: Polish the absolute best found
        if not time_ok() or calls >= budget:
            return best_x, best_f
            
        # If we have budget left, do one more precise run from the best point
        if calls < budget - 5:
            try:
                res = minimize(wrapped_func, best_x, method='L-BFGS-B', 
                               bounds=[(-5.0, 5.0)] * d,
                               options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20})
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
