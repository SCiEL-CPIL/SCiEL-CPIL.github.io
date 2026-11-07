"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Two-phase optimizer: Latin Hypercube Sampling followed by Local Search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, global_best_x, global_best_fx
            if calls_made >= budget:
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline - 10.0

        # Phase 1: Latin Hypercube Sampling (LHS)
        n_lhs = min(50, budget)
        
        def generate_lhs(n, dim, rng):
            points = np.zeros((n, dim))
            for d in range(dim):
                intervals = np.linspace(0, 1, n + 1)
                samples = rng.uniform(intervals[:-1], intervals[1:])
                perm = rng.permutation(n)
                points[:, d] = samples[perm]
            points = points * 10.0 - 5.0
            return points

        if n_lhs > 0:
            lhs_points = generate_lhs(n_lhs, dim, rng)
            for x in lhs_points:
                if not time_ok() or calls_made >= budget:
                    break
                evaluate(x)

        if global_best_x is None:
            x0 = np.zeros(dim)
            evaluate(x0)
            if global_best_x is None:
                return None, np.inf

        # Phase 2: Local Search using SciPy
        x_best = global_best_x.copy()
        
        def scipy_func(x):
            nonlocal calls_made, global_best_x, global_best_fx
            if calls_made >= budget:
                return 1e10
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
            return fx

        try:
            # Use Nelder-Mead which is robust and available
            # maxiter is number of iterations, each uses dim+1 calls
            # We limit it to ensure we don't exceed budget significantly
            # 100 iterations * 6 calls/iter = 600 calls, but budget is 500.
            # So maxiter=50 is safer (300 calls).
            res = minimize(scipy_func, x_best, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xtol': 1e-6, 'ftol': 1e-6})
        except Exception:
            pass

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
