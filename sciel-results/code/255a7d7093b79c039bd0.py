"""Rewritten optimizer using CMA-ES with time budget guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit, use 325s wall-clock kill margin
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_f = np.inf
        num_calls = 0
        
        def check_time():
            return time.time() > deadline - 5  # 5s safety margin
        
        def evaluate(x):
            nonlocal num_calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            num_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Multi-start Nelder-Mead
        n_starts = 10
        points_per_start = 500 // n_starts  # ~50 calls per start
        
        for start_idx in range(n_starts):
            if num_calls >= self.budget or check_time():
                break
            
            # Initialize starting point
            if start_idx == 0:
                x0 = np.zeros(self.dim)
            else:
                # Random starting point within the box
                x0 = rng.uniform(-4, 4, size=self.dim)
            
            # Run Nelder-Mead from this start
            # Use a callback to track calls and check time
            def callback(xk):
                nonlocal num_calls
                num_calls += 1
                if num_calls >= self.budget or check_time():
                    raise RuntimeError("Time or budget exceeded")
            
            try:
                result = minimize(
                    lambda x: evaluate(x),
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': points_per_start,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
            except RuntimeError:
                pass
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
