import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 280.0  # Safety margin before 300s kill

        def time_exceeded():
            return time.time() - start_time > time_limit

        # Hooke-Jeeves pattern search
        # Deterministic, uses function smoothness
        # Alternates exploratory coordinate-wise steps and pattern steps

        # Initial point: center of box
        x = np.zeros(d)
        f = func(x)
        calls = 1
        best_x = x.copy()
        best_f = f

        # Initial step size
        delta = 1.0
        min_delta = 1e-6

        # Pattern step direction (last successful direction)
        pattern_dir = np.zeros(d)

        while calls < budget and not time_exceeded():
            if delta < min_delta:
                break

            # Exploratory search: try each coordinate
            new_x = x.copy()
            improved = False

            for i in range(d):
                # Try positive step
                trial = new_x.copy()
                trial[i] += delta
                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                calls += 1

                if f_trial < f:
                    f = f_trial
                    new_x = trial
                    improved = True
                    if f < best_f:
                        best_f = f
                        best_x = new_x.copy()

                    # Continue in same direction
                    while calls < budget and not time_exceeded():
                        trial2 = new_x.copy()
                        trial2[i] += delta
                        trial2 = np.clip(trial2, -5, 5)
                        f_trial2 = func(trial2)
                        calls += 1
                        if f_trial2 < f:
                            f = f_trial2
                            new_x = trial2
                            if f < best_f:
                                best_f = f
                                best_x = new_x.copy()
                        else:
                            break

                # Try negative step
                trial = new_x.copy()
                trial[i] -= delta
                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                calls += 1

                if f_trial < f:
                    f = f_trial
                    new_x = trial
                    improved = True
                    if f < best_f:
                        best_f = f
                        best_x = new_x.copy()

                    # Continue in same direction
                    while calls < budget and not time_exceeded():
                        trial2 = new_x.copy()
                        trial2[i] -= delta
                        trial2 = np.clip(trial2, -5, 5)
                        f_trial2 = func(trial2)
                        calls += 1
                        if f_trial2 < f:
                            f = f_trial2
                            new_x = trial2
                            if f < best_f:
                                best_f = f
                                best_x = new_x.copy()
                        else:
                            break

            if not improved:
                # No improvement: reduce step size
                delta *= 0.5
                continue

            # Pattern step: try to move in the direction of last improvement
            # The direction is from old x to new_x
            direction = new_x - x
            if np.linalg.norm(direction) > 1e-12:
                pattern_trial = new_x + direction
                pattern_trial = np.clip(pattern_trial, -5, 5)
                f_pattern = func(pattern_trial)
                calls += 1

                if f_pattern < f:
                    # Pattern step succeeded
                    x = new_x.copy()
                    new_x = pattern_trial.copy()
                    f = f_pattern
                    if f < best_f:
                        best_f = f
                        best_x = new_x.copy()

                    # Continue pattern steps
                    while calls < budget and not time_exceeded():
                        pattern_trial2 = new_x + direction
                        pattern_trial2 = np.clip(pattern_trial2, -5, 5)
                        f_pattern2 = func(pattern_trial2)
                        calls += 1
                        if f_pattern2 < f:
                            f = f_pattern2
                            new_x = pattern_trial2
                            if f < best_f:
                                best_f = f
                                best_x = new_x.copy()
                        else:
                            break

                    # After pattern step, set x to new_x and continue exploratory
                    x = new_x.copy()
                else:
                    # Pattern step failed, just update x
                    x = new_x.copy()
            else:
                x = new_x.copy()

            # If we made progress, keep delta; otherwise it's already reduced
            # Actually, if we improved, keep delta the same
            # If we didn't improve, delta was already reduced

        return best_x, best_f
