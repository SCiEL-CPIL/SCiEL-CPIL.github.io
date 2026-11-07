"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic gradient-free trust-region method for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_made = 1

        # Powell's method: maintains a set of conjugate directions
        # Initialize with standard basis directions
        directions = np.eye(n)
        step_sizes = np.ones(n) * 0.5

        max_no_improve = 20
        no_improve = 0
        prev_best = best_f

        while calls_made < budget and time.time() < deadline:
            # Check time
            if time.time() >= deadline:
                break

            improved = False
            min_f = best_f
            min_x = best_x.copy()
            min_dir_idx = -1
            min_step = 0.0

            # Cycle through each direction
            for i in range(n):
                if calls_made >= budget or time.time() >= deadline:
                    break

                # Line search along direction i
                # Try positive and negative steps
                for sign in [1.0, -1.0]:
                    if calls_made >= budget or time.time() >= deadline:
                        break

                    # Try a few step sizes
                    for mult in [1.0, 0.5, 2.0]:
                        if calls_made >= budget or time.time() >= deadline:
                            break

                        step = sign * step_sizes[i] * mult
                        new_x = min_x + step * directions[i]
                        new_x = np.clip(new_x, -5.0, 5.0)
                        new_f = func(new_x)
                        calls_made += 1

                        if new_f < min_f:
                            min_f = new_f
                            min_x = new_x.copy()
                            min_dir_idx = i
                            min_step = step
                            improved = True

            # Update best if improved
            if improved and min_f < best_f:
                best_f = min_f
                best_x = min_x.copy()
                no_improve = 0
            else:
                no_improve += 1

            # If we found an improvement along some direction, update the direction set
            if improved and min_dir_idx >= 0:
                # Powell's direction update:
                # Replace the direction that gave the largest improvement with the net displacement
                # For simplicity, we'll just keep the current directions but adjust step size
                # based on success
                if min_f < prev_best - 1e-10:
                    step_sizes[min_dir_idx] *= 1.2
                else:
                    step_sizes[min_dir_idx] *= 0.8
                step_sizes = np.clip(step_sizes, 1e-4, 10.0)

                # Occasionally rotate directions to maintain diversity
                if no_improve == 0 and calls_made % 50 < n:
                    # Add a small random perturbation to directions to avoid getting stuck
                    # in a degenerate direction set
                    noise = rng.standard_normal(n) * 0.1
                    directions = np.array([d + noise * 0.01 for d in directions])
                    # Re-orthogonalize approximately
                    for j in range(n):
                        for k in range(j + 1, n):
                            dot = np.dot(directions[j], directions[k])
                            directions[k] = directions[k] - dot * directions[j]
                            norms = np.linalg.norm(directions[k])
                            if norms > 1e-10:
                                directions[k] /= norms

            # Check for stagnation
            if no_improve > max_no_improve:
                # Restart with a random perturbation around the current best
                perturbation = rng.standard_normal(n) * 0.5
                new_start = best_x + perturbation
                new_start = np.clip(new_start, -5.0, 5.0)
                new_f = func(new_start)
                calls_made += 1
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_start.copy()
                no_improve = 0
                # Reset directions
                directions = np.eye(n)
                step_sizes = np.ones(n) * 0.5

            prev_best = best_f

            # Early termination if step sizes are too small
            if np.all(step_sizes < 1e-6):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
