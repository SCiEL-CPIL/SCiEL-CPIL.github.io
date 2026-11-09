"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Powell Optimizer with adaptive step-size refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 325.0  # 300s limit, 25s margin
        
        # Wrapper to count calls and enforce limits
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(x)

        best_x = np.zeros(self.dim)
        best_fx = float('inf')
        
        # Try a few initial points
        # 1. Zero (common center for BBOB)
        # 2-5. Random points
        starts = [np.zeros(self.dim)]
        rng = np.random.default_rng(123)
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, self.dim))

        for x0 in starts:
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break
                
            try:
                res = minimize(
                    eval_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 500,
                        'ftol': 1e-12,
                        'xatol': 1e-8
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except StopIteration:
                break
            except Exception:
                # If Powell crashes for some reason, try next start
                continue

        # Final high-precision polish pass if budget allows
        remaining_calls = self.budget - self.calls
        if remaining_calls > 20 and time.time() - self.start_time < self.time_limit - 5:
            try:
                res = minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 100,
                        'ftol': 1e-15,
                        'xatol': 1e-12
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except StopIteration:
                pass
            except Exception:
                pass

        # Ensure we return something valid
        if best_x is None or not np.all(np.isfinite(best_x)):
             best_x = np.zeros(self.dim)
             if self.calls < self.budget:
                 best_fx = eval_func(best_x)
             else:
                 best_fx = 0.0
                 
        return best_x, best_fx
# EVOLVE-BLOCK-END
