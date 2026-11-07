import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315  # 15s safety margin from 330s limit
        calls = 0
        max_calls = self.budget
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time Limit")
            
            calls += 1
            x = np.asarray(x, dtype=float)
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Coarse Exploration with more samples
        n_coarse = 150
        for _ in range(n_coarse):
            x = rng.uniform(-5, 5, self.dim)
            try:
                wrapper(x)
            except (ValueError, TimeoutError):
                break

        # Phase 2: Local Refinement using L-BFGS-B which often works better for smooth functions
        if best_x is not None:
            try:
                minimize(
                    wrapper, 
                    best_x, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 200, 
                        'ftol': 1e-10, 
                        'gtol': 1e-8
                    }
                )
            except Exception:
                pass

            # Phase 3: Further refinement with Nelder-Mead if we still have budget
            if best_x is not None and calls < max_calls and time.time() < deadline:
                try:
                    minimize(
                        wrapper, 
                        best_x, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': 100, 
                            'xatol': 1e-12, 
                            'fatol': 1e-12
                        }
                    )
                except Exception:
                    pass

        if best_x is None:
            # Fallback if something went wrong immediately
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
