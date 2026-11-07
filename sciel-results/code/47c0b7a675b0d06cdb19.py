"""Rewritten optimizer for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, deadline guard, and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        # (1+1)-ES with 1/5 success rule
        sigma = 1.0
        successes = 0
        total = 0
        calls = 1

        for i in range(1, budget):
            if time.time() > deadline:
                break

            # Adapt sigma every 10 steps using 1/5 rule
            if total > 0 and total % 10 == 0:
                rate = successes / total
                if rate > 0.2:
                    sigma *= 1.3
                elif rate < 0.2:
                    sigma *= 1.0 / 1.3
                successes = 0
                total = 0

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            total += 1

        # Final polish: smaller sigma, more iterations if time permits
        if time.time() < deadline - 5.0:
            sigma = 0.1
            successes = 0
            total = 0
            polish_budget = min(100, budget - calls)
            for i in range(polish_budget):
                if time.time() > deadline:
                    break
                if total > 0 and total % 10 == 0:
                    rate = successes / total
                    if rate > 0.2:
                        sigma *= 1.2
                    elif rate < 0.2:
                        sigma *= 1.0 / 1.2
                    successes = 0
                    total = 0

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                total += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
