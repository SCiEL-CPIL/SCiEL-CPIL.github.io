"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES-like optimizer with adaptive step size and refinement."""

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
            return time.time() < deadline

        # Phase 1: Multi-start exploration with (1+1)-ES using 1/5 success rule
        n_starts = 8
        explore_budget = int(budget * 0.55)
        per_start = max(1, explore_budget // n_starts)

        # Diverse starting points
        starts = [
            np.zeros(dim),
            np.ones(dim) * 3.0,
            -np.ones(dim) * 3.0,
            np.ones(dim) * 1.5,
            -np.ones(dim) * 1.5,
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-2, 2, dim),
        ]

        for s in range(n_starts):
            if calls >= budget or not time_ok():
                break

            x = starts[s].copy()
            fx = evaluate(x)
            if not time_ok() or calls >= budget:
                break

            sigma = 2.0
            # 1/5 success rule: track successes in a sliding window
            success_window = []
            window_size = 5
            stagnant = 0

            for _ in range(per_start):
                if calls >= budget or not time_ok():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                    stagnant = 0
                else:
                    success_window.append(0)
                    stagnant += 1

                # Keep window size constant
                if len(success_window) > window_size:
                    success_window.pop(0)

                # Adapt sigma using 1/5 success rule
                if len(success_window) >= window_size:
                    rate = sum(success_window) / window_size
                    if rate > 0.2:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.8
                    else:
                        sigma *= 0.99

                sigma = np.clip(sigma, 0.05, 5.0)

                # Early stop if very stagnant
                if stagnant > 15:
                    break

        # Phase 2: Focused refinement with adaptive sigma
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            success_window = []
            window_size = 3

            for _ in range(budget - calls):
                if calls >= budget or not time_ok():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                else:
                    success_window.append(0)

                if len(success_window) > window_size:
                    success_window.pop(0)

                # Adapt sigma in refinement
                if len(success_window) >= window_size:
                    rate = sum(success_window) / window_size
                    if rate > 0.3:
                        sigma *= 1.15
                    elif rate < 0.1:
                        sigma *= 0.7
                    else:
                        sigma *= 0.95

                sigma = np.clip(sigma, 0.001, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
