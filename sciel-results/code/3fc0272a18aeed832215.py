# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Fast Nelder-Mead with strict deadline guard for BBOB f8 (Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.perf_counter()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        rng = np.random.default_rng(42)

        # Track calls and best
        calls = [0]
        best_x = [None]
        best_f = [np.inf]
        timeout_flag = [False]

        def f(x):
            if time.perf_counter() >= deadline:
                timeout_flag[0] = True
                return np.inf
            calls[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            if val < best_f[0]:
                best_f[0] = val
                best_x[0] = x.copy()
            return val

        # Initial best: a few random starts
        for _ in range(10):
            if calls[0] >= self.budget or time.perf_counter() >= deadline:
                break
            x0 = rng.uniform(-5.0, 5.0, self.dim)
            f(x0)

        if best_x[0] is None:
            return np.zeros(self.dim), np.inf

        # Multi-start Nelder-Mead
        # Nelder-Mead is robust and fast for dim=5.
        # We use adaptive=True to handle scaling issues.
        # We limit maxfev to remaining budget.
        
        # Generate a few diverse starting points
        start_points = [best_x[0].copy()]
        
        # Add points around the best
        for scale in [1.0, 0.5, 2.0]:
            if calls[0] >= self.budget or time.perf_counter() >= deadline:
                break
            x0 = np.clip(best_x[0] + scale * rng.standard_normal(self.dim), -5.0, 5.0)
            f(x0)
            start_points.append(best_x[0].copy())
            
        # Add random spread
        for _ in range(5):
            if calls[0] >= self.budget or time.perf_counter() >= deadline:
                break
            x0 = rng.uniform(-5.0, 5.0, self.dim)
            f(x0)
            start_points.append(best_x[0].copy())

        # Run Nelder-Mead from each start point
        for x0 in start_points:
            if calls[0] >= self.budget or time.perf_counter() >= deadline:
                break
            
            # Calculate remaining budget for this run
            remaining_budget = self.budget - calls[0]
            if remaining_budget <= 0:
                break
            
            # Cap maxfev to a reasonable number to avoid long runs
            # Nelder-Mead in dim 5 needs ~ (dim+1) * iterations
            # Let's allow up to 200 function evals per start
            maxfev = min(remaining_budget, 200)
            
            try:
                # Use a simple wrapper to catch timeout
                def f_wrapper(x):
                    if timeout_flag[0]:
                        raise ValueError("Timeout")
                    return f(x)
                    
                res = scipy.optimize.minimize(
                    f_wrapper, x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100, 
                        'maxfev': maxfev,
                        'xatol': 1e-8, 
                        'fatol': 1e-10, 
                        'adaptive': True
                    },
                )
            except Exception:
                # If timeout or other error, break out
                if timeout_flag[0]:
                    break
                continue

            if best_x[0] is not None:
                # Add the new best to start points for further polishing?
                # No, just proceed to next start or final polish.
                pass

        # Final polish if budget and time remain
        if calls[0] < self.budget and best_x[0] is not None and time.perf_counter() < deadline:
            remaining_budget = self.budget - calls[0]
            maxfev = min(remaining_budget, 100)
            
            try:
                def f_wrapper(x):
                    if timeout_flag[0]:
                        raise ValueError("Timeout")
                    return f(x)

                res = scipy.optimize.minimize(
                    f_wrapper, best_x[0],
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50, 
                        'maxfev': maxfev,
                        'xatol': 1e-10, 
                        'fatol': 1e-12, 
                        'adaptive': True
                    },
                )
            except Exception:
                pass

        # Final safety: ensure we return something
        if best_x[0] is None:
            best_x[0] = np.zeros(self.dim)
            best_f[0] = float('inf')

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
