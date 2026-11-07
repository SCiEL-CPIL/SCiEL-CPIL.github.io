"""Hooke-Jeeves pattern search for the BBOB f6 (Attractive sector) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Hooke-Jeeves pattern search with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0
        n = self.dim
        budget = self.budget

        # Start at origin
        x = np.zeros(n)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        # Initial step size
        step = 1.0
        min_step = 1e-8

        # Pattern search state
        last_success_dir = None  # direction of last successful step

        while calls < budget and time.time() < deadline:
            # Phase 1: Exploratory moves (coordinate-wise)
            x_base = x.copy()
            f_base = fx
            found_improvement = False
            success_dir = None

            for i in range(n):
                if calls >= budget or time.time() >= deadline:
                    break

                # Try positive step in dimension i
                x_trial = x_base.copy()
                x_trial[i] += step
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls += 1

                if f_trial < f_base:
                    x_base = x_trial
                    f_base = f_trial
                    found_improvement = True
                    success_dir = np.zeros(n)
                    success_dir[i] = 1.0
                else:
                    # Try negative step in dimension i
                    if calls >= budget or time.time() >= deadline:
                        break

                    x_trial = x.copy()
                    x_trial[i] -= step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1

                    if f_trial < f_base:
                        x_base = x_trial
                        f_base = f_trial
                        found_improvement = True
                        success_dir = np.zeros(n)
                        success_dir[i] = -1.0

            if found_improvement:
                x = x_base
                fx = f_base
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Phase 2: Pattern move (accelerate along last success direction)
                if success_dir is not None:
                    # Try to move further in the same direction
                    pattern_x = x + step * success_dir
                    pattern_x = np.clip(pattern_x, -5, 5)
                    if calls < budget and time.time() < deadline:
                        pattern_f = func(pattern_x)
                        calls += 1

                        if pattern_f < fx:
                            # Success: keep accelerating
                            while calls < budget and time.time() < deadline:
                                next_x = x + 2 * step * success_dir
                                next_x = np.clip(next_x, -5, 5)
                                next_f = func(next_x)
                                calls += 1

                                if next_f < pattern_f:
                                    x = next_x
                                    fx = next_f
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()
                                    pattern_f = next_f
                                else:
                                    break
                            # After pattern move, restart exploratory from new x
                            # and double the step size (or keep it)
                            step = min(step * 2.0, 5.0)  # accelerate
                        else:
                            # Pattern move failed, halve step
                            step = step * 0.5
                    else:
                        step = step * 0.5
                else:
                    # No clear direction, halve step
                    step = step * 0.5
            else:
                # No improvement in exploratory phase
                step = step * 0.5
                if step < min_step:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
