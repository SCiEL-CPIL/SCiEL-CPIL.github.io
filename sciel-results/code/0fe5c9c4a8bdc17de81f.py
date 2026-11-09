"""Optimizer for BBOB-style Katsuura f23 with AOCC-aware search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: stratified exploration then (1+1)-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s limit
        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        # Phase 1: Stratified sampling for quick broad exploration
        # Use Latin Hypercube-like sampling for diversity
        n_explore = min(80, budget // 6)
        best_x = None
        best_f = np.inf

        for i in range(n_explore):
            if time.time() > deadline:
                break
            # Stratified: divide each dimension into bins, pick one per bin
            x = np.zeros(dim)
            for d in range(dim):
                # Use a stratified position based on index
                frac = (i + 0.5) / n_explore
                # Add randomness within the stratum
                x[d] = -5.0 + 10.0 * (frac + rng.uniform(-0.5, 0.5) / n_explore)
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES from the best point found
        x = best_x if best_x is not None else rng.uniform(-4, 4, dim)
        fx = best_f if best_f < np.inf else func(x)

        sigma = 1.0
        success_count = 0
        total_steps = 0
        window_size = 15
        successes_in_window = 0

        for step in range(budget - n_explore):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            total_steps += 1

            if fy < fx:
                x, fx = y, fy
                successes_in_window += 1
                # Increase sigma on success
                sigma *= 1.2
            else:
                # Decrease sigma on failure
                sigma *= 0.9

            # Adaptive sigma adjustment based on success rate in window
            if total_steps % window_size == 0:
                success_rate = successes_in_window / window_size
                if success_rate > 0.5:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma *= 0.7
                successes_in_window = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.001, 5.0)

        return x, fx
# EVOLVE-BLOCK-END
