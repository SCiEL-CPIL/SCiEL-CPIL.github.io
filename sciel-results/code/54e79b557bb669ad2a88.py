"""Multi-resolution coordinate descent with adaptive step halving for BBOB f7."""
import numpy as np
import time


class Optimizer:
    """Multi-Resolution Coordinate Descent with Adaptive Step Halving."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        def time_left():
            return deadline - time.time()

        # Phase 1: Sparse grid initialization to identify promising basins
        # Evaluate a small set of structured points to find a good starting region
        best_x = np.zeros(n)
        best_fx = np.inf
        eval_count = 0

        # Grid points: center, corners, and some intermediate points
        grid_points = []
        # Center
        grid_points.append(np.zeros(n))
        # Axis-aligned points at various scales
        for scale in [0.0, 1.0, 2.0, 3.0, 4.0]:
            for i in range(n):
                for sign in [1, -1]:
                    pt = np.zeros(n)
                    pt[i] = sign * scale
                    grid_points.append(pt)
        # Random scatter points
        for _ in range(15):
            grid_points.append(rng.uniform(-4, 4, n))

        # Evaluate grid points, keep best
        for pt in grid_points:
            if time_left() < 5.0:
                break
            pt_clipped = np.clip(pt, -5, 5)
            fx = func(pt_clipped)
            eval_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = pt_clipped.copy()

        # Phase 2: Coordinate descent with adaptive step halving
        # For each coordinate, try moving in both directions and shrink step if no improvement
        step_sizes = np.array([4.0] * n)
        min_step = 1e-4
        no_improve_count = 0
        max_no_improve = 50

        while eval_count < budget and time_left() > 5.0:
            improved = False

            for i in range(n):
                if eval_count >= budget or time_left() < 5.0:
                    break

                step = step_sizes[i]
                if step < min_step:
                    continue

                # Try positive direction
                x_trial = best_x.copy()
                x_trial[i] = np.clip(x_trial[i] + step, -5, 5)
                fx_trial = func(x_trial)
                eval_count += 1

                if fx_trial < best_fx - 1e-12:
                    best_fx = fx_trial
                    best_x = x_trial.copy()
                    improved = True
                    # Keep or slightly increase step
                    step_sizes[i] = min(step * 1.05, 5.0)
                else:
                    # Try negative direction
                    x_trial = best_x.copy()
                    x_trial[i] = np.clip(x_trial[i] - step, -5, 5)
                    fx_trial = func(x_trial)
                    eval_count += 1

                    if fx_trial < best_fx - 1e-12:
                        best_fx = fx_trial
                        best_x = x_trial.copy()
                        improved = True
                        step_sizes[i] = min(step * 1.05, 5.0)
                    else:
                        # No improvement: halve the step size
                        step_sizes[i] = step * 0.5

            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    # Restart from a random point near best
                    if eval_count < budget - 10 and time_left() > 10.0:
                        x_new = best_x + rng.normal(0, 0.5, n)
                        x_new = np.clip(x_new, -5, 5)
                        fx_new = func(x_new)
                        eval_count += 1
                        if fx_new < best_fx:
                            best_fx = fx_new
                            best_x = x_new.copy()
                            no_improve_count = 0
                            step_sizes = np.array([2.0] * n)
                        else:
                            # Try a completely random restart
                            if eval_count < budget - 5:
                                x_new = rng.uniform(-4, 4, n)
                                fx_new = func(x_new)
                                eval_count += 1
                                if fx_new < best_fx:
                                    best_fx = fx_new
                                    best_x = x_new.copy()
                                    no_improve_count = 0
                                    step_sizes = np.array([3.0] * n)
                    else:
                        break
            else:
                no_improve_count = 0

        return best_x, best_fx
