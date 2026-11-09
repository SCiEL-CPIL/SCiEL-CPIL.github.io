"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using Powell's Method with adaptive step sizes for BBOB f14.
    Combines coordinate-wise line search with direction updates to exploit
    the quasi-separable structure of Different Powers.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        calls = 0

        def _eval(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            x_clipped = np.clip(x, -5.0, 5.0)
            calls += 1
            return func(x_clipped)

        # Initial point: origin is a good heuristic for shifted/rotated BBOB
        x = np.zeros(dim)
        fx = _eval(x)
        best_x = x.copy()
        best_fx = fx

        # Powell's method: maintain a set of search directions
        directions = np.eye(dim).astype(float)
        # Per-direction step sizes, initialized to 1.0
        step_sizes = np.ones(dim)

        # Tolerances
        tol_f = 1e-12
        tol_x = 1e-10

        # Main Powell loop
        max_powell_cycles = 200

        for cycle in range(max_powell_cycles):
            if calls >= budget or time.time() > deadline:
                break

            x_cycle_start = x.copy()
            f_cycle_start = fx

            # Track improvement per direction for the direction update
            dir_improvements = np.zeros(dim)
            dir_best_points = [None] * dim
            dir_best_fxs = [None] * dim

            # Search along each direction
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break

                d = directions[i]
                step = step_sizes[i]

                # 1D minimization along direction d using adaptive stepping
                # Try to find a better point along this line
                x_best = x.copy()
                f_best = fx
                t_best = 0.0

                # First, determine direction of movement
                # Try positive step
                x_trial = x + step * d
                f_trial = _eval(x_trial)

                if f_trial < f_best:
                    # Move in positive direction, try to extend
                    x_best = x_trial.copy()
                    f_best = f_trial
                    t_best = step
                    # Try extending further
                    while calls < budget and time.time() < deadline:
                        x_try = x_best + step * d
                        f_try = _eval(x_try)
                        if f_try < f_best:
                            x_best = x_try
                            f_best = f_try
                            t_best += step
                            # Increase step for this direction
                            step_sizes[i] = min(step * 1.3, 10.0)
                        else:
                            break
                else:
                    # Try negative step
                    x_trial = x - step * d
                    f_trial = _eval(x_trial)
                    if f_trial < f_best:
                        # Move in negative direction, try to extend
                        x_best = x_trial.copy()
                        f_best = f_trial
                        t_best = -step
                        # Try extending further
                        while calls < budget and time.time() < deadline:
                            x_try = x_best - step * d
                            f_try = _eval(x_try)
                            if f_try < f_best:
                                x_best = x_try
                                f_best = f_try
                                t_best -= step
                                step_sizes[i] = min(step * 1.3, 10.0)
                            else:
                                break
                    else:
                        # No improvement; shrink step size
                        step_sizes[i] *= 0.7
                        if step_sizes[i] < 1e-6:
                            step_sizes[i] = 1e-6

                # Record improvement for this direction
                if f_best < fx:
                    x = x_best
                    fx = f_best
                    dir_improvements[i] = fx_cycle_start - fx if False else f_cycle_start - f_best
                    # Actually, improvement relative to start of this direction search
                    dir_improvements[i] = fx - f_best  # positive means improvement
                    dir_best_points[i] = x_best.copy()
                    dir_best_fxs[i] = f_best

                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx

            # End of direction loop

            # Powell's direction update:
            # Compare f at cycle start vs end
            f_cycle_end = fx
            improvement = f_cycle_start - f_cycle_end

            if improvement > tol_f and calls < budget and time.time() < deadline:
                # Find the direction with the least improvement
                # If all directions improved, replace the worst one
                if np.any(dir_improvements > 0):
                    worst_dir = np.argmin(dir_improvements)
                else:
                    worst_dir = 0

                # Check if the overall move is better than the worst direction
                # The new direction should be from x_cycle_start to x
                new_dir = x - x_cycle_start
                new_dir_norm = np.linalg.norm(new_dir)
                if new_dir_norm > tol_x:
                    new_dir = new_dir / new_dir_norm

                    # Check if new_dir is too similar to any existing direction
                    replace = True
                    for j in range(dim):
                        if j == worst_dir:
                            continue
                        cos_sim = np.dot(new_dir, directions[j]) / (new_dir_norm * np.linalg.norm(directions[j]) + 1e-15)
                        if cos_sim > 0.999:
                            replace = False
                            break

                    if replace:
                        directions[worst_dir] = new_dir
                        # Reset step size for the new direction
                        step_sizes[worst_dir] = 1.0
                        # Try a search along the new direction
                        x_trial = x + step_sizes[worst_dir] * new_dir
                        f_trial = _eval(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            if fx < best_fx:
                                best_x = x.copy()
                                best_fx = fx
                        else:
                            x_trial = x - step_sizes[worst_dir] * new_dir
                            f_trial = _eval(x_trial)
                            if f_trial < fx:
                                x = x_trial
                                fx = f_trial
                                if fx < best_fx:
                                    best_x = x.copy()
                                    best_fx = fx

            # Check convergence
            if abs(improvement) < tol_f:
                # Try a few more cycles with smaller steps
                step_sizes *= 0.5
                if np.all(step_sizes < 1e-8):
                    break

        # Final polish: fine coordinate search around best_x
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            fine_steps = [0.05, 0.01, 0.002, 0.0005]
            for step in fine_steps:
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[d] += direction * step
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        f_trial = _eval(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            if fx < best_fx:
                                best_x = x.copy()
                                best_fx = fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
