import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        
        def check_deadline():
            return time.time() < deadline - 2.0

        # --- Phase 1: Initial Sampling ---
        # Use a larger portion of the budget for global search to ensure we find the global basin.
        # f11 has a central optimum, but shifted. 100 points in 5D is a good start.
        n_init = 100
        rng = np.random.default_rng(12345)
        
        points = []
        for i in range(n_init):
            if not check_deadline() or eval_count >= self.budget:
                break
            x = rng.uniform(-5.0, 5.0, self.dim)
            fx = func(x)
            eval_count += 1
            points.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if eval_count >= self.budget or not check_deadline():
            return best_x, best_f

        # Sort points by fitness
        points.sort(key=lambda x: x[0])
        
        # --- Phase 2: Local Refinement using Nelder-Mead ---
        # Pick the top K unique-ish points to refine.
        K = 3
        candidates = []
        for fx, x in points[:K]:
            candidates.append(x.copy())

        remaining = self.budget - eval_count
        
        # Define a wrapper for func to count evaluations and check deadline
        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= self.budget or not check_deadline():
                return best_f # Return a bad value or current best to stop
            fx = func(np.clip(x, -5.0, 5.0))
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.clip(x, -5.0, 5.0).copy()
            return fx

        # Run Nelder-Mead on each candidate
        for x0 in candidates:
            if eval_count >= self.budget or not check_deadline():
                break
            
            try:
                # maxiter is not the exact eval count, but a limit on iterations.
                # Nelder-Mead typically uses ~dim * maxiter evaluations.
                # We want to use the remaining budget.
                # Let's set a generous maxiter but rely on the wrapper to stop.
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            except Exception:
                pass

        return best_x, best_f
