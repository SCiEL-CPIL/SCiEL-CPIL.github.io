"""Multi-start local search with coordinate refinement and adaptive step sizes."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start coordinate descent with adaptive step sizes and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        time_limit = 325.0

        best_x = None
        best_f = np.inf
        eval_count = 0

        # Number of restarts: partition budget into chunks
        # We'll do adaptive restarts: each restart uses ~8-15 evals
        restarts = 30
        evals_per_restart = max(5, budget // restarts)

        def time_left():
            return (time_limit - (time.time() - start_time)) > 0.5

        def check_time():
            return (time.time() - start_time) < time_limit

        for r in range(restarts):
            if eval_count >= budget - 2 or not check_time():
                break

            # Choose starting point
            if r == 0:
                # First restart: center
                x = np.zeros(n)
            elif r == 1:
                x = np.random.uniform(-4, 4, n)
            elif r == 2:
                x = np.random.uniform(-4, 4, n) * 0.5
            elif r < 6:
                # Structured points
                for coord in range(n):
                    pass
                # Use a grid-like starting point
                x = np.random.uniform(-3, 3, n)
                # Make one coordinate more extreme
                x[np.random.randint(n)] = np.random.uniform(-4.5, 4.5)
            else:
                # Random restarts with some structure
                if r % 3 == 0:
                    x = np.random.uniform(-4, 4, n)
                elif r % 3 == 1:
                    x = np.random.uniform(-2, 2, n)
                else:
                    x = np.random.uniform(-5, 5, n) * 0.7

            x = np.clip(x, -5, 5)
            fx = func(x)
            eval_count += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            if eval_count >= budget or not check_time():
                break

            # Local search: coordinate ascent/descent with adaptive step
            # Try coordinate-wise moves with multiple step sizes
            base_steps = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]

            # For each coordinate, try positive and negative moves
            # Use a greedy approach: try large steps first, then refine
            improved = True
            while improved and eval_count < budget and check_time():
                improved = False
                for coord in range(n):
                    if eval_count >= budget or not check_time():
                        break
                    for step in base_steps:
                        if eval_count >= budget or not check_time():
                            break
                        # Try positive move
                        x_trial = x.copy()
                        x_trial[coord] = np.clip(x[coord] + step, -5, 5)
                        if not np.allclose(x_trial, x):
                            f_trial = func(x_trial)
                            eval_count += 1
                            if f_trial < fx - 1e-10:
                                x = x_trial
                                fx = f_trial
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                improved = True
                                break  # Found improvement, try other coords
                        # Try negative move
                        if eval_count >= budget or not check_time():
                            break
                        x_trial = x.copy()
                        x_trial[coord] = np.clip(x[coord] - step, -5, 5)
                        if not np.allclose(x_trial, x):
                            f_trial = func(x_trial)
                            eval_count += 1
                            if f_trial < fx - 1e-10:
                                x = x_trial
                                fx = f_trial
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                improved = True
                                break  # Found improvement, try other coords

            # If no improvement, try a small random perturbation to escape
            if not improved and eval_count < budget - 1 and check_time():
                x_trial = np.clip(x + np.random.standard_normal(n) * 0.3, -5, 5)
                f_trial = func(x_trial)
                eval_count += 1
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
