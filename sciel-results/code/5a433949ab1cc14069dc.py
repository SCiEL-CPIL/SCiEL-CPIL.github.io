"""Optimized continuous optimizer for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: CMA-ES-like global search + local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls = 0

        def safe_func(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            return float(func(x))

        rng = np.random.default_rng(42)

        # Phase 1: Global search with adaptive step size
        x = rng.uniform(-5, 5, dim)
        fx = safe_func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        stall = 0
        stall_limit = 40

        # Track progress for AOCC
        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = safe_func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
                stall = 0
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                sigma *= 0.92
                stall += 1
                # If stuck for a while, do a larger jump
                if stall > stall_limit:
                    sigma = max(sigma, 1.5)
                    stall = 0

        # Phase 2: Local refinement around best point
        if calls < budget - dim:
            # Coordinate descent with fine steps
            local_sigma = max(0.01, sigma * 0.1)
            for coord in range(dim):
                for _ in range(5):
                    if calls >= budget - dim or time.time() > deadline:
                        break
                    direction = np.zeros(dim)
                    direction[coord] = 1.0
                    y_plus = np.clip(best_x + local_sigma * direction, -5, 5)
                    y_minus = np.clip(best_x - local_sigma * direction, -5, 5)
                    fp = safe_func(y_plus)
                    fm = safe_func(y_minus)
                    if fp < best_f:
                        best_x = y_plus.copy()
                        best_f = fp
                        local_sigma *= 1.2
                    elif fm < best_f:
                        best_x = y_minus.copy()
                        best_f = fm
                        local_sigma *= 1.2
                    else:
                        local_sigma *= 0.7

        # Phase 3: Final local search with even finer steps
        if calls < budget - dim:
            local_sigma = max(0.001, local_sigma * 0.1)
            for coord in range(dim):
                for _ in range(3):
                    if calls >= budget - dim or time.time() > deadline:
                        break
                    direction = np.zeros(dim)
                    direction[coord] = 1.0
                    y_plus = np.clip(best_x + local_sigma * direction, -5, 5)
                    y_minus = np.clip(best_x - local_sigma * direction, -5, 5)
                    fp = safe_func(y_plus)
                    fm = safe_func(y_minus)
                    if fp < best_f:
                        best_x = y_plus.copy()
                        best_f = fp
                    elif fm < best_f:
                        best_x = y_minus.copy()
                        best_f = fm
                    else:
                        local_sigma *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
