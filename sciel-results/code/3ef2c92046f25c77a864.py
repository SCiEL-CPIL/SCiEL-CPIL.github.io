"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size, time guard, and early local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        total_budget = self.budget
        dim = self.dim

        # Multi-restart strategy for Rastrigin (many local minima)
        # 6 restarts of ~83 evaluations each
        n_restarts = 6
        per_restart = total_budget // n_restarts
        remaining = total_budget - per_restart * n_restarts

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def check_time():
            return time.time() < deadline - 3.0

        def check_budget():
            return calls_made < total_budget

        for r in range(n_restarts):
            if not check_time() or not check_budget():
                break

            # Initialize: first restart at origin, subsequent random but center-biased
            if r == 0:
                x = np.zeros(dim)
            else:
                # Biased towards center to exploit Rastrigin structure
                x = np.random.uniform(-3, 3, dim)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES with adaptive sigma
            sigma = 1.5
            iters = per_restart if r < n_restarts - 1 else per_restart + remaining

            for i in range(iters):
                if not check_time() or not check_budget():
                    break

                # Generate candidate with Gaussian perturbation
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 0.005, 8.0)

            if not check_budget():
                break

        # Final local polish around the best solution found
        if best_x is not None and check_time() and check_budget():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            while check_time() and check_budget():
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
                sigma = np.clip(sigma, 0.001, 3.0)

        if best_x is None:
            # Fallback: should not happen, but ensure we return something
            x = np.zeros(dim)
            best_f = func(x)
            calls_made += 1
            best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
