import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """(1+1)-ES with adaptive sigma and Nelder-Mead polish for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Track total calls
        total_calls = 0

        def tracked_func(x):
            nonlocal total_calls
            total_calls += 1
            return func(x)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = tracked_func(x)

        best_x = x.copy()
        best_fx = fx
        sigma = 1.0

        # Reserve last 80 calls for Nelder-Mead polish
        polish_budget = 80
        es_budget = budget - polish_budget

        # (1+1)-ES with simple sigma adaptation
        for _ in range(es_budget - 1):
            if time.time() >= deadline:
                break
            if total_calls >= budget:
                break

            # Generate candidate
            z = rng.standard_normal(dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = tracked_func(y)

            if fy < fx:
                x, fx = y, fy
                # Increase sigma on success (moderate)
                sigma *= 1.2
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                # Decrease sigma on failure (moderate)
                sigma *= 0.9

            # Clamp sigma
            if sigma > 5.0:
                sigma = 5.0
            elif sigma < 1e-10:
                sigma = 1e-10

        # Phase 2: Nelder-Mead polish using remaining budget
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline:
            nm_calls = [0]

            def nm_func(z):
                nonlocal total_calls
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                total_calls += 1
                return func(z)

            # Use Nelder-Mead from best point
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
