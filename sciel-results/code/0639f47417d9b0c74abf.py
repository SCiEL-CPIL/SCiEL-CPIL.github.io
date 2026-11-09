"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """
    Multi-stage derivative-free optimizer:
    1. Random sampling for initial exploration.
    2. Nelder-Mead local refinements from diverse starting points.
    3. (1+1)-ES polish with adaptive step size.
    
    Uses a time guard to stay within the 300s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        rng = np.random.default_rng(42)
        calls_used = 0
        best_x = None
        best_f = np.inf

        def check_time():
            nonlocal start_time, deadline
            return (time.time() - start_time) > (deadline - 5.0)

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Random sampling for broad exploration
        n_random = min(50, self.budget)
        for _ in range(n_random):
            if calls_used >= self.budget or check_time():
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            calls_used += 1
            try_update(x, f)

        if calls_used >= self.budget or check_time():
            return best_x, best_f

        # Phase 2: Nelder-Mead local refinements
        # Start from best points found in Phase 1
        n_nm = min(10, (self.budget - calls_used) // 50)
        
        # Select diverse starting points
        # Use the best point and some random points near it
        start_points = [best_x.copy()]
        for _ in range(min(n_nm - 1, 5)):
            if calls_used >= self.budget or check_time():
                break
            # Perturb the best point
            perturbation = rng.normal(0, 1.5, self.dim)
            x_start = np.clip(best_x + perturbation, -5, 5)
            start_points.append(x_start)

        for x_start in start_points:
            if calls_used >= self.budget or check_time():
                break
            
            # Track local calls
            local_calls = [0]
            
            def nm_func(x):
                nonlocal calls_used, local_calls
                if local_calls[0] >= 40 or calls_used >= self.budget:
                    return best_f  # Return current best to stop
                x_clipped = np.clip(x, -5, 5)
                f = func(x_clipped)
                calls_used += 1
                local_calls[0] += 1
                return f

            try:
                result = minimize(
                    nm_func,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    try_update(result.x, result.fun)
            except Exception:
                pass

        if calls_used >= self.budget or check_time():
            return best_x, best_f

        # Phase 3: (1+1)-ES polish
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        
        while calls_used < self.budget and not check_time():
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                try_update(x, fx)
                sigma *= 1.3
            else:
                sigma *= 0.85
            
            if sigma < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
