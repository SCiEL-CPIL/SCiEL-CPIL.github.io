"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B with time guard for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)

        # Track best solution
        best_x = None
        best_f = np.inf

        # Count function calls
        call_count = 0

        # Wrapper to count calls and track best
        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Random sampling to find good starting points
        n_random = min(50, budget // 5)
        for _ in range(n_random):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = wrapped_func(x)

        # Phase 2: L-BFGS-B optimization from multiple starts
        # Use the best random point and a few others
        starts = [best_x.copy()]
        
        # Add some diverse starting points
        for _ in range(4):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            starts.append(x)

        for x0 in starts:
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 1000, 'maxfun': budget - call_count, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: If budget remains, do a fine-tuning (1+1)-ES
        if call_count < budget and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while call_count < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = wrapped_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
