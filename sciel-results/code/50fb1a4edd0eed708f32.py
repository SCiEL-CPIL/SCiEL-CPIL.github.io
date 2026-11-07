"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hooke-Jeeves pattern search with coordinate-wise line searches and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_made = 0

        def check_deadline():
            return time.time() < deadline

        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        calls_made += 1
        best_x = x.copy()
        best_fx = fx

        # Initial step sizes: start moderate
        delta = np.full(dim, 1.0)
        min_delta = 1e-8
        max_delta = 10.0
        max_coord_steps = 50  # max steps per coordinate line search
        max_outer_iters = budget - 1

        outer_iter = 0
        while outer_iter < max_outer_iters and calls_made < budget:
            if not check_deadline():
                break

            improved = False
            # Coordinate-wise line search (expanding step)
            for i in range(dim):
                if not check_deadline() or calls_made >= budget:
                    break

                step_count = 0
                while step_count < max_coord_steps:
                    # Try positive direction
                    y = x.copy()
                    y[i] += delta[i]
                    if y[i] > 5.0:
                        break
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        improved = True
                        step_count += 1
                        if calls_made >= budget:
                            break
                        continue
                    else:
                        # Try negative direction
                        y = x.copy()
                        y[i] -= delta[i]
                        if y[i] < -5.0:
                            break
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                            step_count += 1
                            if calls_made >= budget:
                                break
                            continue
                        else:
                            # No improvement in either direction, shrink step
                            delta[i] *= 0.5
                            if delta[i] < min_delta:
                                delta[i] = min_delta
                            break

                # If we made progress in this coordinate, keep or grow step
                if improved and step_count > 0:
                    delta[i] = min(delta[i] * 1.5, max_delta)

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # If no coordinate improved, shrink all steps
            if not improved:
                delta *= 0.5
                if np.all(delta < min_delta):
                    break

            outer_iter += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
