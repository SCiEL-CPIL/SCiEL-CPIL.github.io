"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and coordinate-aware search for bent cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize at center (0,0,...,0) which is often near the optimum for shifted functions
        x = np.zeros(dim)
        fx = func(x)
        best_fx = fx
        best_x = x.copy()
        calls = 1

        # Initial sigma
        sigma = 1.0

        # Track per-dimension progress to detect ill-conditioning
        # For bent cigar, one dimension dominates. We'll adaptively adjust sigma per dimension.
        sigma_vec = np.ones(dim)

        # Success window for global sigma adaptation
        success_window = []
        window_size = 15

        # Per-dimension success tracking
        dim_success = np.zeros(dim)
        dim_window_size = 10

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * sigma_vec * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            # Check for improvement
            if fy < fx:
                # Success
                success_window.append(1)
                for d in range(dim):
                    if abs(y[d] - x[d]) > 1e-10:
                        dim_success[d] += 1
                x, fx = y, fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                success_window.append(0)

            # Update global sigma using 1/5 rule
            if len(success_window) >= window_size:
                if len(success_window) > window_size:
                    success_window.pop(0)
                success_rate = sum(success_window) / window_size
                if success_rate > 0.3:
                    sigma *= 1.3
                elif success_rate < 0.2:
                    sigma *= 0.8

            # Update per-dimension sigma
            for d in range(dim):
                if dim_success[d] > dim_window_size * 0.4:
                    sigma_vec[d] *= 1.2
                elif dim_success[d] < dim_window_size * 0.1:
                    sigma_vec[d] *= 0.8
                # Decay dim_success
                dim_success[d] *= 0.9

            # Clamp sigma_vec to reasonable range
            sigma_vec = np.clip(sigma_vec, 0.01, 10.0)

            # Occasionally try coordinate-wise exploration if stuck
            if calls % 50 == 0 and sigma < 0.1:
                # Try a larger step in one random dimension
                d = rng.integers(dim)
                step = rng.uniform(0.5, 2.0) * (1 if rng.random() < 0.5 else -1)
                y2 = x.copy()
                y2[d] += step
                y2 = np.clip(y2, -5, 5)
                fy2 = func(y2)
                calls += 1
                if fy2 < fx:
                    x, fx = y2, fy2
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
