"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


class Optimizer:
    """Differential Evolution optimizer for Katsuura function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget or time.time() > deadline - 1:
                raise RuntimeError("Budget or time exceeded")
            calls_made += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)
            return f

        bounds = [(-5.0, 5.0)] * self.dim

        # Run differential evolution with a large population and many generations
        # DE is population-based and handles multimodality well
        # maxiter * (popsize + 1) evaluations approximately
        # We want to use most of the 500 budget
        # Let's use popsize=50, maxiter=8 -> 50*8 + 50 = 450 evals
        # Or popsize=30, maxiter=14 -> 30*14 + 30 = 450
        # Let's try popsize=40, maxiter=10 -> 40*10 + 40 = 440
        
        try:
            res = differential_evolution(
                counted_func,
                bounds,
                maxiter=10,
                popsize=40,
                tol=1e-8,
                seed=42,
                polish=False,  # We'll do our own polish
                init='sobol'
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except RuntimeError:
            pass
        except Exception:
            pass

        # If we still have budget and time, do a local polish with Nelder-Mead
        if best_x is not None and calls_made < self.budget - 10 and time.time() < deadline - 10:
            from scipy.optimize import minimize
            remaining = self.budget - calls_made
            maxiter_nm = max(1, (remaining // 6) - 2)
            try:
                res2 = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter_nm, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x
            except RuntimeError:
                pass
            except Exception:
                pass

        # Fallback: if no calls were made
        if calls_made == 0:
            x0 = np.random.uniform(-5, 5, self.dim)
            best_f = func(x0)
            best_x = x0
            calls_made = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
