"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


class Optimizer:
    """Lattice search + adaptive (1+1)-ES for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin

        def elapsed_ok():
            return (time.time() - start_time) < deadline

        best_x = None
        best_fx = np.inf
        calls_used = 0

        # Phase 1: Deterministic 3^5 lattice search over [-5, 5]^5
        grid_vals = [-5.0, 0.0, 5.0]
        grid_points = [np.array(combo) for combo in itertools.product(grid_vals, repeat=n)]

        for x in grid_points:
            if calls_used >= budget - 1 or not elapsed_ok():
                break
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Refined lattice around the best grid point (5^1 = 5 points per dim, but use 3^5 again with smaller step)
        # Actually, let's do a finer local lattice: step size 1.0 around best_x, 3^5 = 243 more points
        if calls_used < budget - 1 and elapsed_ok():
            # Fine lattice: centered at best_x, step = 1.0, 3 points per dim -> 3^5 = 243
            step = 1.0
            fine_grid_vals = [[best_x[i] - step, best_x[i], best_x[i] + step] for i in range(n)]
            # Generate all combinations
            for combo in itertools.product(*fine_grid_vals):
                if calls_used >= budget - 1 or not elapsed_ok():
                    break
                x = np.clip(np.array(combo), -5.0, 5.0)
                fx = func(x)
                calls_used += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        # Phase 3: Adaptive (1+1)-ES from the best point
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5

        # Adaptive sigma tracking
        successes = 0
        total_steps = 0
        window = 15

        while calls_used < budget - 1 and elapsed_ok():
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                sigma *= 1.03
            else:
                sigma *= 0.97

            total_steps += 1
            if total_steps >= window:
                rate = successes / total_steps
                if rate > 0.25:
                    sigma *= 1.15
                elif rate < 0.1:
                    sigma *= 0.85
                successes = 0
                total_steps = 0

            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 5.0)

        # Phase 4: Final polish with very small steps
        if calls_used < budget - 1 and elapsed_ok():
            sigma = 0.005
            polish_steps = min(30, budget - 1 - calls_used)
            for _ in range(polish_steps):
                if not elapsed_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy

        return x, fx
# EVOLVE-BLOCK-END
