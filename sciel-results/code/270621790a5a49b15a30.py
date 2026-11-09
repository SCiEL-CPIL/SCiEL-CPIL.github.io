"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with adaptive cooling for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        # Initialize from broad uniform scout
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Simulated Annealing parameters
        T = 10.0  # initial temperature
        cooling_rate = 0.997  # slow cooling
        min_T = 1e-6
        sigma = 1.5  # step size

        # Track recent history for stagnation detection
        no_improve_count = 0
        stagnation_threshold = 30

        for i in range(budget - 1):
            # Deadline check
            if time.time() > deadline:
                break

            # Adaptive step size: shrink when stuck, expand when making progress
            if no_improve_count > stagnation_threshold:
                sigma *= 1.5
                no_improve_count = 0
            elif no_improve_count > 0:
                sigma *= 0.95

            # Generate candidate
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            # Acceptance criterion
            if fy < fx:
                # Always accept improvements
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = y.copy(), fx
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Accept worse solutions with probability exp(-delta/T)
                delta = fy - fx
                if delta < 0:
                    accept = True
                else:
                    accept = rng.random() < np.exp(-delta / T)

                if accept:
                    x, fx = y, fy
                    no_improve_count += 1
                else:
                    no_improve_count += 1

            # Cool down
            T = max(T * cooling_rate, min_T)

        return best_x, best_fx
# EVOLVE-BLOCK-END
