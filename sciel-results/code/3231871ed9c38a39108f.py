# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf
        self._start_time = None

    def __call__(self, func):
        self._start_time = time.time()
        deadline = self._start_time + 280  # Safety margin for 300s limit
        
        # Wrapper to count function calls and enforce budget
        def wrapped_func(x):
            if self._call_count >= self.budget:
                raise ValueError("Budget exceeded")
            self._call_count += 1
            # Ensure x is a valid vector within bounds for the optimizer
            # L-BFGS-B handles bounds, but let's be safe
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            return val

        # Generate diverse starting points
        # 1. Center
        starts = [np.zeros(self.dim)]
        
        # 2. Axis-aligned points near boundaries
        for i in range(self.dim):
            for val in [-4.0, 4.0]:
                p = np.zeros(self.dim)
                p[i] = val
                starts.append(p)
        
        # 3. Random points
        rng = np.random.default_rng(42)
        n_random = 10
        random_starts = rng.uniform(-4.5, 4.5, size=(n_random, self.dim))
        starts.extend(random_starts.tolist())
        
        # Sort starts by some heuristic? No, just iterate.
        # We will run L-BFGS-B from each start.
        # To prevent any single run from consuming all budget, 
        # we estimate maxfun per run.
        # Budget is 500. We have ~20 starts. 
        # Let's give each run a max of 20 function evaluations.
        # L-BFGS-B is efficient, so this should be enough to find the basin.
        
        max_fun_per_run = 20
        
        for start_point in starts:
            if self._call_count >= self.budget:
                break
            if time.time() > deadline - 5: # 5s buffer
                break
                
            try:
                res = minimize(
                    wrapped_func, 
                    start_point, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxfun': max_fun_per_run, 'maxiter': 100, 'disp': False}
                )
                
                if res.success or res.fun < self._best_f:
                    if res.fun < self._best_f:
                        self._best_f = res.fun
                        self._best_x = res.x
            except Exception:
                # If an exception occurs, we keep the calls made so far
                # and continue with the next start point if budget allows
                pass

        # If we have remaining budget and time, we can do a final polish
        # using a small ES around the best point found, or just return.
        # Given the nature of f7 (flat minimum), L-BFGS-B should have landed 
        # in the flat region if it found the basin.
        
        if self._best_x is None:
             # Fallback if no valid result (should not happen)
             self._best_x = np.zeros(self.dim)
             self._best_f = func(self._best_x)
             self._call_count += 1

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
