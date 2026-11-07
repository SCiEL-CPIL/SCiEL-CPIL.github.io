import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290
        
        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return np.inf
            return func(x)

        # 1. Global Exploration with LHS
        # 50 points in 5D is a reasonable start
        n_lhs = 50
        sampler = LatinHypercube(d=dim, seed=42)
        lhs_points = sampler.random(n=n_lhs)
        # Scale to [-4, 4]
        lhs_points = lhs_points * 8.0 - 4.0

        lhs_results = []
        for x in lhs_points:
            if call_count[0] >= budget - 100: # Reserve budget
                break
            f_val = wrapped_func(x)
            lhs_results.append((f_val, x))

        if not lhs_results:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)
            return best_x, best_f

        lhs_results.sort(key=lambda item: item[0])
        best_f, best_x = lhs_results[0]

        # 2. Local Refinement
        # Use top 3 points as seeds for local optimization
        top_k = 3
        starts = [p[1] for p in lhs_results[:top_k]]
        
        bounds = [(-5.0, 5.0)] * dim
        
        for x0 in starts:
            if time.time() > deadline:
                break
            if call_count[0] >= budget - 20:
                break

            # L-BFGS-B is efficient for smooth bounded problems
            # Rosenbrock is smooth.
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

            # If we have budget left, try SLSQP for a polish on the current best
            if time.time() < deadline and call_count[0] < budget - 10:
                try:
                    res2 = minimize(
                        wrapped_func,
                        best_x,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': 50, 'ftol': 1e-12, 'disp': False}
                    )
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.x
                except Exception:
                    pass

        return best_x, best_f
