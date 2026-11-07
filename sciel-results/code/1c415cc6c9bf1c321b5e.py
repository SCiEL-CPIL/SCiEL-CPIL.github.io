"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        # Multi-start with different seeds for diversity
        rng = np.random.default_rng(42)

        # Phase 1: Multi-start exploration
        # We'll do several short runs from different starting points
        num_starts = 10
        calls_per_start = max(1, (self.budget - 50) // num_starts)

        best_x = None
        best_fx = np.inf

        for start_idx in range(num_starts):
            if time.time() > deadline:
                break

            # Different starting points
            if start_idx == 0:
                x = np.zeros(self.dim)
            elif start_idx == 1:
                x = rng.uniform(-4, 4, self.dim)
            elif start_idx == 2:
                x = rng.uniform(-5, 5, self.dim)
            elif start_idx == 3:
                x = np.array([rng.uniform(-4, 4) for _ in range(self.dim)])
            else:
                # Perturb the current best
                if best_x is not None:
                    x = np.clip(best_x + rng.normal(0, 1.5, self.dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # Adaptive (1+1)-ES from this start
            sigma = 1.5
            calls_in_start = 0
            max_calls_in_start = calls_per_start

            while calls_in_start < max_calls_in_start:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_in_start += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.85

                # If sigma becomes too small, break
                if sigma < 1e-6:
                    break

        # Phase 2: Fine-tuning around the best point
        if best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1

            remaining_time = deadline - time.time()
            # Estimate how many more calls we can make
            # Assume ~1ms per call, so ~1000 calls/sec
            max_fine_calls = min(100, int(remaining_time * 50))

            for _ in range(max_fine_calls):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9

                if sigma < 1e-8:
                    break

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
