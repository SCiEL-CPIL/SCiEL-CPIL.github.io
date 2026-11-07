"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with CMA-like step adaptation, multiple restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        best_x = np.zeros(dim)
        best_f = float('inf')

        # Phase 1: single (1+1)-ES from origin for most of budget
        x = np.zeros(dim)
        fx = func(x)
        if fx < best_f:
            best_f, best_x = fx, x.copy()

        sigma = 3.0
        success_count = 0
        fail_count = 0
        window = 10
        calls_made = 1
        max_phase1 = budget - 50  # reserve 50 calls for polish

        for i in range(max_phase1 - 1):
            if time.time() > deadline:
                break
            calls_made += 1
            z = rng.standard_normal(dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)

            if fy < fx:
                # Success
                x, fx = y, fy
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                # Step-size increase
                sigma *= 1.15
                success_count += 1
                fail_count = 0
            else:
                sigma *= 0.85
                fail_count += 1

            # If stuck for too long, make a larger exploratory jump
            if fail_count >= 15:
                jump_scale = 5.0
                y2 = np.clip(x + sigma * jump_scale * rng.standard_normal(dim), -5, 5)
                fy2 = func(y2)
                calls_made += 1
                if fy2 < fx:
                    x, fx = y2, fy2
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.1
                fail_count = 0

        # Phase 2: local polish with very small sigma
        x = best_x.copy()
        fx = best_f
        sigma = 0.05
        remaining = budget - calls_made
        if remaining > 0 and time.time() < deadline:
            for i in range(remaining):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma = min(sigma * 1.1, 0.5)
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
