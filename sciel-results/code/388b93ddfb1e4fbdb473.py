import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
                start_time = time.time()
                deadline = start_time + 290  # 10s safety margin
                calls = 0
                max_calls = self.budget
                best_x = None
                best_f = np.inf
                rng = np.random.default_rng(42) # Fixed seed for reproducibility? Or random?
                # The task says "hidden seeded shift". It doesn't say our seed is fixed.
                # Using a fixed seed for our internal RNG is fine.

                def wrapper(x):
                    nonlocal calls, best_x, best_f
                    if calls >= max_calls:
                        raise ValueError("Budget Exceeded")
                    if time.time() > deadline:
                        raise TimeoutError("Time Limit")
                    
                    calls += 1
                    # Ensure x is a numpy array
                    x = np.asarray(x, dtype=float)
                    val = func(x)
                    
                    if val < best_f:
                        best_f = val
                        best_x = x.copy()
                    return val

                # Phase 1: Coarse Exploration
                n_coarse = 100
                for _ in range(n_coarse):
                    x = rng.uniform(-5, 5, self.dim)
                    try:
                        wrapper(x)
                    except (ValueError, TimeoutError):
                        break

                # Phase 2: Local Refinement
                if best_x is not None:
                    try:
                        # Nelder-Mead
                        # maxiter is just a cap, wrapper handles budget
                        minimize(
                            wrapper, 
                            best_x, 
                            method='Nelder-Mead', 
                            options={
                                'maxiter': 500, 
                                'xatol': 1e-10, 
                                'fatol': 1e-10
                            }
                        )
                    except Exception:
                        pass # Ignore budget/time exceptions, we have best_x

                if best_x is None:
                    # Fallback if something went wrong immediately
                    best_x = np.zeros(self.dim)
                    best_f = func(best_x)

                return best_x, best_f
