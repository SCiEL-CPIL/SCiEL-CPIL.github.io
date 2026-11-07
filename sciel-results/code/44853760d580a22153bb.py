import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """(1+1)-ES with adaptive sigma and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Track total function calls
        total_calls = [0]

        def tracked_func(x):
            total_calls[0] += 1
            return func(x)

        # Phase 1: (1+1)-ES with adaptive sigma and restarts
        # Reserve last 60 calls for Nelder-Mead polish
        polish_budget = 60
        es_budget = budget - polish_budget

        best_x = None
        best_fx = np.inf
        sigma = 3.0
        no_improve_count = 0
        max_no_improve = 80  # Restart after this many non-improving steps

        x = rng.uniform(-5, 5, dim)
        fx = tracked_func(x)
        best_x = x.copy()
        best_fx = fx

        for step in range(es_budget - 1):
            if time.time() >= deadline:
                break
            if total_calls[0] >= es_budget:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = tracked_func(y)

            if fy < fx:
                # Step accepted
                x, fx = y, fy
                # Adaptive sigma: increase on success, but more conservatively
                sigma *= 1.3
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Step rejected
                sigma *= 0.8
                no_improve_count += 1

            # Clamp sigma
            if sigma > 8.0:
                sigma = 8.0
            elif sigma < 1e-10:
                sigma = 1e-10

            # Restart if stuck: move to best_x with fresh sigma
            if no_improve_count >= max_no_improve:
                x = best_x + 0.5 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = tracked_func(x)
                sigma = max(sigma, 1.0)
                no_improve_count = 0

        # Phase 2: Nelder-Mead polish using remaining budget
        remaining = budget - total_calls[0]
        if remaining > 3 and time.time() < deadline:
            nm_calls = [0]

            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                return func(z)

            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'maxfev': remaining
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
