"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global grid exploration followed by (1+1)-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Track best point and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            return fx

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        # Phase 1: Global grid exploration
        # Use a modest grid to find the basin of attraction
        # With 5D and budget 500, we can afford a small grid
        # Grid size: 3^5 = 243 points, leaving ~257 for local search
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            pt = np.array([
                                -4.0 + i * 4.0,  # -4, 0, 4
                                -4.0 + j * 4.0,
                                -4.0 + k * 4.0,
                                -4.0 + l * 4.0,
                                -4.0 + m * 4.0,
                            ])
                            grid_points.append(pt)

        # Shuffle grid points to avoid bias
        rng = np.random.default_rng(42)
        order = rng.permutation(len(grid_points))

        for idx in order:
            if time.time() > deadline - 120.0:
                break
            if calls_made >= budget - 150:  # reserve budget for local search
                break
            pt = grid_points[idx]
            fx = safe_call(pt)
            update_best(pt, fx)

        # Phase 2: Local (1+1)-ES refinement from the best grid point
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0

            # Adaptive (1+1)-ES with 1/5 success rule
            successes = 0
            trials = 0

            while calls_made < budget and time.time() < deadline - 5.0:
                y = x + sigma * rng.standard_normal(dim)
                fy = safe_call(y)
                trials += 1

                if fy < fx:
                    x = y.copy()
                    fx = fy
                    update_best(x, fy)
                    successes += 1
                    sigma *= 1.1
                else:
                    sigma *= 0.9

                # Adapt sigma based on success rate over last 10 trials
                if trials % 10 == 0:
                    rate = successes / trials
                    if rate > 0.3:
                        sigma *= 1.2
                    elif rate < 0.1:
                        sigma *= 0.7
                    successes = 0
                    trials = 0

                # If sigma becomes too small, try a larger jump
                if sigma < 0.01:
                    sigma = 0.5

        # Phase 3: If budget remains, do a few more random restarts
        remaining = budget - calls_made
        if remaining > 20 and time.time() < deadline - 10.0:
            for _ in range(min(remaining, 30)):
                if time.time() > deadline - 5.0:
                    break
                x_try = rng.uniform(-5, 5, dim)
                fx_try = safe_call(x_try)
                update_best(x_try, fx_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
