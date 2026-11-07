"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pattern search with adaptive step size and diagonal directions for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim

        # Initialize at origin or a random point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Build direction set: axes + diagonals
        directions = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
            directions.append(-d)
        # Add diagonal directions (all +1, all -1, and pairwise)
        for _ in range(4):
            d = rng.choice([-1.0, 1.0], size=dim)
            if np.any(d != 0):
                directions.append(d / np.linalg.norm(d))
        # Add some random directions
        for _ in range(6):
            d = rng.standard_normal(dim)
            d /= np.linalg.norm(d)
            directions.append(d)

        # Adaptive step size
        step = 1.0
        min_step = 1e-10
        max_step = 10.0

        while calls < self.budget:
            # Check time
            if time.time() > deadline:
                break

            improved = False
            for d in directions:
                if calls >= self.budget:
                    break
                if time.time() > deadline:
                    break

                x_try = np.clip(x + step * d, -5.0, 5.0)
                fx_try = func(x_try)
                calls += 1

                if fx_try < fx:
                    x = x_try
                    fx = fx_try
                    improved = True
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    # Expand step when improvement
                    step = min(step * 1.5, max_step)
                    break
                else:
                    # Try negative direction
                    x_try_neg = np.clip(x - step * d, -5.0, 5.0)
                    fx_try_neg = func(x_try_neg)
                    calls += 1

                    if fx_try_neg < fx:
                        x = x_try_neg
                        fx = fx_try_neg
                        improved = True
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        step = min(step * 1.5, max_step)
                        break

            if not improved:
                # Shrink step
                step *= 0.5
                if step < min_step:
                    # Try to restart with a new random direction set
                    # Generate fresh random directions
                    for i in range(len(directions)):
                        d = rng.standard_normal(dim)
                        norm = np.linalg.norm(d)
                        if norm > 0:
                            directions[i] = d / norm
                    step = 1.0

                    # If still stuck, try a small random perturbation
                    if step < min_step * 10:
                        # Do a local random search
                        for _ in range(min(10, self.budget - calls)):
                            if time.time() > deadline:
                                break
                            x_try = np.clip(x + 0.1 * rng.standard_normal(dim), -5.0, 5.0)
                            fx_try = func(x_try)
                            calls += 1
                            if fx_try < fx:
                                x = x_try
                                fx = fx_try
                                if fx < best_fx:
                                    best_fx = fx
                                    best_x = x.copy()

                    if step < min_step * 0.1:
                        break

        # Final polish: fine-tune around best point
        if calls < self.budget:
            fine_step = 1e-3
            for _ in range(min(50, self.budget - calls)):
                if time.time() > deadline:
                    break
                d = rng.standard_normal(dim)
                d /= np.linalg.norm(d)
                x_try = np.clip(best_x + fine_step * d, -5.0, 5.0)
                fx_try = func(x_try)
                calls += 1
                if fx_try < best_fx:
                    best_fx = fx_try
                    best_x = x_try.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
