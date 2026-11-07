"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate Descent with adaptive per-dimension step sizes for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        rng = np.random.default_rng(42)

        # Track best seen so far
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal best_x, best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        calls_used = 0

        # Phase 1: Initial exploration with several diverse starting points
        n_init = 5
        init_points = []
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = eval_point(x)
            calls_used += 1
            init_points.append((x, fx))
            if time.time() > deadline:
                break

        init_points.sort(key=lambda t: t[1])

        # Phase 2: Coordinate Descent from the best initial point
        if calls_used < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f

            # Adaptive step sizes per dimension
            steps = np.ones(dim) * 1.0
            min_step = 0.01
            max_step = 5.0

            stagnation_count = 0
            max_stagnation = 30

            while calls_used < budget and time.time() < deadline:
                improved_this_round = False

                for d in range(dim):
                    if calls_used >= budget or time.time() > deadline:
                        break

                    # Try moving in positive direction with line search
                    best_local_f = fx
                    best_local_x = x.copy()
                    trial_step = steps[d]

                    for scale in [1.0, 0.5, 0.25, 0.1, 2.0, 4.0]:
                        if calls_used >= budget or time.time() > deadline:
                            break
                        trial = x.copy()
                        trial[d] = np.clip(x[d] + trial_step * scale, -5, 5)
                        f_trial = eval_point(trial)
                        calls_used += 1
                        if f_trial < best_local_f:
                            best_local_f = f_trial
                            best_local_x = trial.copy()
                            # Update step size based on improvement magnitude
                            if abs(trial_step * scale) > steps[d]:
                                steps[d] = min(steps[d] * 1.3, max_step)
                            else:
                                steps[d] = max(steps[d] * 0.8, min_step)

                    if best_local_x is not x and not np.allclose(best_local_x, x):
                        x = best_local_x
                        fx = best_local_f
                        improved_this_round = True

                    # Try moving in negative direction with line search
                    best_local_f_neg = fx
                    best_local_x_neg = x.copy()
                    trial_step_neg = steps[d]

                    for scale in [1.0, 0.5, 0.25, 0.1, 2.0, 4.0]:
                        if calls_used >= budget or time.time() > deadline:
                            break
                        trial = x.copy()
                        trial[d] = np.clip(x[d] - trial_step_neg * scale, -5, 5)
                        f_trial = eval_point(trial)
                        calls_used += 1
                        if f_trial < best_local_f_neg:
                            best_local_f_neg = f_trial
                            best_local_x_neg = trial.copy()
                            if abs(trial_step_neg * scale) > steps[d]:
                                steps[d] = min(steps[d] * 1.3, max_step)
                            else:
                                steps[d] = max(steps[d] * 0.8, min_step)

                    if best_local_x_neg is not x and not np.allclose(best_local_x_neg, x):
                        x = best_local_x_neg
                        fx = best_local_f_neg
                        improved_this_round = True

                if not improved_this_round:
                    stagnation_count += 1
                    if stagnation_count >= max_stagnation:
                        break
                else:
                    stagnation_count = 0

        # Return best found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
