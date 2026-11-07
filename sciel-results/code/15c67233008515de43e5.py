"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma, stall detection, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        total_calls = 0

        # Multi-restart strategy
        # Use more restarts for multi-modal functions
        num_restarts = min(6, max(2, budget // 80))
        budget_per_restart = budget // num_restarts

        for restart_idx in range(num_restarts):
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize
            if restart_idx == 0:
                # First restart: try center and a few random points
                x = np.zeros(dim)
                fx = func(x)
                total_calls += 1
                if fx < best_fx:
                    best_fx, best_x = fx, x.copy()

                # Try a few random starting points
                for _ in range(min(3, budget - total_calls)):
                    if time.time() > deadline:
                        break
                    x_rand = np.random.default_rng(42 + restart_idx).uniform(-5, 5, dim)
                    fx_rand = func(x_rand)
                    total_calls += 1
                    if fx_rand < best_fx:
                        best_fx, best_x = fx_rand, x_rand.copy()
                    if fx_rand < fx:
                        x, fx = x_rand, fx_rand
            else:
                # Subsequent restarts: start near best or random
                if np.random.rand() < 0.5:
                    x = best_x + np.random.default_rng(42 + restart_idx).standard_normal(dim) * 1.0
                    x = np.clip(x, -5, 5)
                else:
                    x = np.random.default_rng(42 + restart_idx).uniform(-5, 5, dim)
                fx = func(x)
                total_calls += 1
                if fx < best_fx:
                    best_fx, best_x = fx, x.copy()

            # (1+1)-ES with adaptive sigma
            sigma = 2.0
            stall = 0
            threshold = 20
            local_budget = min(budget_per_restart, budget - total_calls)

            for _ in range(local_budget):
                if total_calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate candidate
                y = np.clip(x + sigma * np.random.default_rng(42 + restart_idx).standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    stall = 0
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()
                else:
                    sigma *= 0.9
                    stall += 1

                if stall > threshold:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
