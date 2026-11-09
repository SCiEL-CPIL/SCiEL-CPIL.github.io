import numpy as np
import time
from scipy.optimize import minimize, dual_annealing

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Keep a few top points to use as restarts for local optimization
        top_points = []
        top_fvals = []

        def safe_func(x):
            nonlocal calls, best_x, best_f, top_points, top_fvals
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                raise TimeoutError()
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if not np.isfinite(fx):
                fx = float('inf')
            
            # Update best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            # Update top 5 points
            if len(top_fvals) < 5:
                top_points.append(x.copy())
                top_fvals.append(fx)
                # Keep sorted
                idx = np.argsort(top_fvals)
                top_fvals = [top_fvals[i] for i in idx]
                top_points = [top_points[i] for i in idx]
            else:
                if fx < top_fvals[-1]:
                    top_points[-1] = x.copy()
                    top_fvals[-1] = fx
                    idx = np.argsort(top_fvals)
                    top_fvals = [top_fvals[i] for i in idx]
                    top_points = [top_points[i] for i in idx]
                    
            return fx

        # Phase 1: Global Exploration with Dual Annealing
        # Reduce budget for DA to save calls for local refinement
        da_budget = int(budget * 0.3) # 30% = 150 calls
        if da_budget > 20:
            try:
                bounds = [(-4.0, 4.0)] * dim
                da_result = dual_annealing(
                    safe_func,
                    bounds,
                    maxiter=50,
                    maxfev=da_budget,
                    seed=42,
                    restarts=3
                )
            except Exception:
                pass

        if not np.isfinite(best_f):
            # Fallback
            rng = np.random.default_rng(123)
            for _ in range(min(50, budget - 10)):
                if time.time() > deadline:
                    break
                x = rng.uniform(-4, 4, dim)
                safe_func(x)
            if not np.isfinite(best_f):
                return best_x, best_f

        # Phase 2: Local Refinement with L-BFGS-B from multiple starts
        # Use the top points found in Phase 1
        local_budget_per_start = 40
        max_starts = min(5, (budget - calls) // local_budget_per_start)
        
        # Ensure we have unique starts
        starts = []
        for p in top_points:
            if len(starts) >= max_starts:
                break
            # Check if distinct from existing starts
            is_distinct = True
            for s in starts:
                if np.linalg.norm(p - s) < 1e-3:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append(p)
                
        # If we didn't get enough distinct points, add perturbations
        rng = np.random.default_rng(42)
        while len(starts) < max_starts:
            pert = best_x + rng.normal(0, 0.5, dim)
            pert = np.clip(pert, -4.0, 4.0)
            starts.append(pert)

        for x0 in starts:
            if calls >= budget - 20:
                break
            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfun': local_budget_per_start, 'gtol': 1e-10}
                )
            except Exception:
                pass

        # Phase 3: Final Polish with Powell
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 50, 'maxfev': remaining, 'xtol': 1e-12, 'ftol': 1e-12}
                )
            except Exception:
                pass

        return best_x, best_f
