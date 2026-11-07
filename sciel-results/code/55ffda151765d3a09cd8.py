"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with multiple restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        total_budget = self.budget
        dim = self.dim

        # Multiple restarts: 8 restarts, each with its own (1+1)-ES
        n_restarts = 8
        budget_per_restart = max(1, (total_budget - n_restarts) // n_restarts)
        # Reserve some budget for a final polish
        polish_budget = 50
        budget_per_restart = max(1, (total_budget - n_restarts - polish_budget) // n_restarts)

        best_x = None
        best_fx = float('inf')

        for r in range(n_restarts):
            # Check deadline
            if time.time() > deadline:
                break

            # Different seed per restart for diversity
            seed = 42 + r * 1000
            rng = np.random.default_rng(seed)

            # Start from a diverse point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)

            if fx < best_fx:
                best_fx, best_x = fx, x.copy()

            # (1+1)-ES for this restart
            sigma = 2.0
            sigma_min = 1e-6
            sigma_max = 5.0
            calls_used = 1  # initial evaluation

            for i in range(budget_per_restart):
                if time.time() > deadline:
                    break
                if calls_used >= total_budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.3, sigma_max)
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()
                else:
                    sigma = max(sigma * 0.85, sigma_min)

        # Final polish: use a very small sigma around the best point
        if best_x is not None and time.time() < deadline and calls_used < total_budget:
            rng = np.random.default_rng(12345)
            x = best_x.copy()
            fx = best_fx
            sigma = 0.01
            polish_calls = 0
            for i in range(polish_budget):
                if time.time() > deadline:
                    break
                if calls_used + polish_calls >= total_budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                polish_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.95
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()
                else:
                    sigma = max(sigma * 0.9, 1e-8)

        if best_x is None:
            # Fallback: at least one call
            x = np.zeros(dim)
            best_fx = func(x)
            best_x = x

        return best_x, best_fx
# EVOLVE-BLOCK-END
