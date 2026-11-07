import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = time.time()
        self.time_limit = 300.0

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            self.call_count += 1
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
            x = np.clip(x, -5.0, 5.0)
            return func(x)

        def update_best(x, f):
            if f < self.best_f:
                self.best_f = f
                self.best_x = np.clip(x, -5.0, 5.0).copy()

        n = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0) for _ in range(n)]

        # Phase 1: Latin Hypercube Sampling for diverse global coverage
        # LHS provides better coverage than Halton for this dimension
        n_starts = 30
        
        def latin_hypercube(n_points, dim):
            points = np.zeros((n_points, dim))
            for d in range(dim):
                # Generate random permutation of intervals
                perm = np.random.permutation(n_points)
                # Random offset within each interval
                offsets = np.random.random(n_points)
                # Scale to [0, 1)
                points[:, d] = (perm + offsets) / n_points
            # Scale to [-5, 5]
            points = points * 10.0 - 5.0
            return points

        X0 = latin_hypercube(n_starts, n)
        
        # Evaluate initial points
        candidates = []
        for x in X0:
            if self.call_count >= budget or time.time() - self.start_time > self.time_limit:
                break
            try:
                f = wrapped_func(x)
                candidates.append((f, x.copy()))
                update_best(x, f)
            except Exception:
                pass

        if not candidates:
            return self.best_x, self.best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Powell's method for gradient-free local refinement
        # Powell can escape shallow local minima more effectively than L-BFGS-B
        remaining = budget - self.call_count
        
        top_k = min(5, len(candidates))
        if top_k > 0:
            calls_per_opt = max(15, remaining // top_k)
            
            for i in range(top_k):
                if self.call_count >= budget or time.time() - self.start_time > self.time_limit:
                    break
                
                f0, x0 = candidates[i]
                
                try:
                    res = minimize(wrapped_func, x0, method='Powell', bounds=bounds,
                                   options={'maxiter': 100, 'maxfev': calls_per_opt})
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # Phase 3: Final polish with L-BFGS-B on the absolute best
        if self.call_count < budget - 5 and self.best_x is not None:
            try:
                res = minimize(wrapped_func, self.best_x, method='L-BFGS-B', bounds=bounds,
                               options={'maxfun': budget - self.call_count - 2, 'maxiter': 200})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        return self.best_x, self.best_f
