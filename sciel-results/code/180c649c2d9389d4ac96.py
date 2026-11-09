# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Robust optimizer using Nelder-Mead with random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline - 5.0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                raise Exception("Budget exceeded")
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial exploration
        for _ in range(min(10, budget)):
            if not time_ok() or calls_made >= budget:
                break
            x_try = rng.uniform(-4, 4, dim)
            try:
                evaluate(x_try)
            except Exception:
                break

        if calls_made >= budget or not time_ok():
            return best_x, best_f

        # Random restarts with Nelder-Mead
        # We use a small number of restarts to ensure we explore the space
        # Nelder-Mead is local, so restarts help with global exploration
        n_restarts = 5
        
        for _ in range(n_restarts):
            if not time_ok() or calls_made >= budget:
                break
            
            # Start from a random point or the current best
            if rng.random() < 0.5:
                x0 = rng.uniform(-4, 4, dim)
            else:
                x0 = best_x.copy()
                
            # Evaluate start point
            try:
                evaluate(x0)
            except Exception:
                break
                
            if calls_made >= budget or not time_ok():
                break

            # Run Nelder-Mead
            # maxiter is a loose limit; we rely on the budget exception
            try:
                res = minimize(evaluate, x0, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
                # If it finishes naturally, check if it improved
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                # Budget exceeded or other error, stop
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
