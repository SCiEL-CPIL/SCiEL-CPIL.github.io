"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate descent with periodic restarts and line-search-free steps."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def time_ok():
            return time.time() < deadline

        # Phase 1: Multi-start coordinate descent
        n_starts = 10
        budget_per_start = self.budget // n_starts

        for start_idx in range(n_starts):
            if calls_used >= self.budget or not time_ok():
                break

            # Initialize starting point
            if start_idx == 0:
                x = rng.uniform(-5, 5, self.dim)
            else:
                # Mix of random and perturbed best
                if rng.random() < 0.3 and best_x is not None:
                    x = best_x + rng.normal(0, 1.5, self.dim)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # Coordinate descent with adaptive step size
            step_size = 2.0
            no_improve_count = 0
            max_no_improve = 15

            for coord in range(self.dim * 8):  # 8 cycles through all dimensions
                if calls_used >= self.budget or not time_ok():
                    break

                axis = coord % self.dim
                old_val = x[axis]

                # Try positive and negative steps
                for sign in [1, -1]:
                    if calls_used >= self.budget or not time_ok():
                        break

                    new_val = np.clip(old_val + sign * step_size, -5, 5)
                    if new_val == old_val:
                        continue

                    x[axis] = new_val
                    fy = func(x)
                    calls_used += 1

                    if fy < fx:
                        fx = fy
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                        no_improve_count = 0
                        step_size *= 1.3
                    else:
                        x[axis] = old_val
                        no_improve_count += 1
                        step_size *= 0.75

                # Decay step size if no improvement
                if no_improve_count >= max_no_improve:
                    step_size *= 0.5
                    no_improve_count = 0
                    if step_size < 0.01:
                        break

            # Periodic random perturbation to escape local minima
            if start_idx % 3 == 2 and best_x is not None:
                x = best_x + rng.normal(0, 0.5, self.dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f, best_x = fx, x.copy()

        # Phase 2: Fine-grained local search around best point
        if best_x is not None and calls_used < self.budget and time_ok():
            x = best_x.copy()
            fx = best_f
            step_size = 0.3
            for _ in range(200):
                if calls_used >= self.budget or not time_ok():
                    break
                for axis in range(self.dim):
                    if calls_used >= self.budget or not time_ok():
                        break
                    for sign in [1, -1]:
                        if calls_used >= self.budget or not time_ok():
                            break
                        new_val = np.clip(x[axis] + sign * step_size, -5, 5)
                        if new_val == x[axis]:
                            continue
                        x[axis] = new_val
                        fy = func(x)
                        calls_used += 1
                        if fy < fx:
                            fx = fy
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
                            step_size *= 1.1
                        else:
                            x[axis] -= sign * step_size
                            step_size *= 0.85
                if step_size < 0.001:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
