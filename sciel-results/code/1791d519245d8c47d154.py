"""Optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        # Track best-so-far for AOCC
        def evaluate(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            return fx

        # Phase 1: Multiple random restarts with (1+1)-ES
        # Use a fraction of budget per restart, with increasing exploration
        num_restarts = 5
        budget_per_restart = max(10, (budget - num_restarts * 10) // num_restarts)

        for restart in range(num_restarts):
            if calls_made >= budget:
                break
            if time.time() >= deadline - 5.0:
                break

            # Initialize: for first restart use uniform, later use perturbation of best
            if restart == 0 or best_x is None:
                x = rng.uniform(-5, 5, dim)
            else:
                # Perturb around best found so far
                sigma_init = 2.0 / (restart + 1)
                x = np.clip(best_x + sigma_init * rng.standard_normal(dim), -5, 5)

            fx = evaluate(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            sigma = 1.0
            # Adaptive step size with 1/5 rule
            success_count = 0
            window = 10

            for _ in range(budget_per_restart):
                if calls_made >= budget:
                    break
                if time.time() >= deadline - 2.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    pass

                # Adapt sigma based on success rate
                if success_count == 0 and _ > 0:
                    sigma *= 0.85
                elif _ >= window:
                    rate = success_count / window
                    if rate > 0.5:
                        sigma *= 1.2
                    elif rate < 0.2:
                        sigma *= 0.8
                    success_count = 0

        # Phase 2: Final polish with smaller steps around best
        if best_x is not None and calls_made < budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            remaining = budget - calls_made
            for _ in range(remaining):
                if time.time() >= deadline - 1.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    best_x = x.copy()
                    best_fx = fx
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
