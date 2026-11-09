"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and multi-start."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin from 300s

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return calls < budget and time.time() < deadline

        # Phase 1: Multi-start exploration with CMA-ES-like adaptation
        # Use ~60% of budget for exploration with 4 different starts
        n_starts = 4
        explore_budget = int(budget * 0.60)
        per_start = explore_budget // n_starts

        for s in range(n_starts):
            if not time_ok():
                break

            # Different starting points
            if s == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = evaluate(x)
            sigma = 2.0
            # Success counter for 1/5 rule
            successes = 0
            total_steps = 0
            window = 5

            for _ in range(per_start):
                if not time_ok():
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                total_steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1

                # 1/5 success rule: adapt sigma every `window` steps
                if total_steps % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.2:
                        sigma *= 1.15
                    elif success_rate < 0.2:
                        sigma *= 0.85
                    successes = 0

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.05, 5.0)

        # Phase 2: Dedicated exploitation with smaller sigma
        if best_x is not None and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            refine_budget = budget - calls

            successes = 0
            total_steps = 0
            window = 5

            for _ in range(refine_budget):
                if not time_ok():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                total_steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1

                if total_steps % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.85
                    successes = 0

                sigma = np.clip(sigma, 0.005, 2.0)

        # Phase 3: Final local refinement with very small steps
        if best_x is not None and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            final_budget = budget - calls

            for _ in range(final_budget):
                if not time_ok():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9

                sigma = max(sigma, 0.001)

        return best_x, best_f
# EVOLVE-BLOCK-END
