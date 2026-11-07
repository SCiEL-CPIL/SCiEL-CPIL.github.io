"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate Descent with adaptive per-axis step sizes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(12345)

        # Track best
        best_x = np.zeros(dim)
        best_f = np.inf

        # Per-axis adaptive step sizes
        steps = np.ones(dim) * 2.0
        # Per-axis direction memory (sign of last successful move)
        last_dirs = np.ones(dim)

        def eval_point(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = eval_point(x)
        calls = 1

        # Coordinate descent with adaptive steps
        while calls < budget:
            if time.time() > deadline:
                break

            improved_this_round = False

            for d in range(dim):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Try a step in the current best direction first
                trial = x.copy()
                trial[d] += steps[d] * last_dirs[d]
                trial = np.clip(trial, -5.0, 5.0)
                f_trial = func(trial)
                calls += 1

                if f_trial < fx:
                    # Accept, update best direction
                    x = trial
                    fx = f_trial
                    improved_this_round = True
                    # Increase step slightly (but cap it)
                    steps[d] = min(steps[d] * 1.2, 4.0)
                    # Keep same direction
                else:
                    # Try opposite direction
                    trial2 = x.copy()
                    trial2[d] -= steps[d] * 0.5
                    trial2 = np.clip(trial2, -5.0, 5.0)
                    f_trial2 = func(trial2)
                    calls += 1

                    if f_trial2 < fx:
                        x = trial2
                        fx = f_trial2
                        improved_this_round = True
                        # Flip direction
                        last_dirs[d] *= -1
                        # Shrink step
                        steps[d] = max(steps[d] * 0.7, 0.05)
                    else:
                        # No improvement; shrink step and try smaller
                        steps[d] = max(steps[d] * 0.8, 0.01)
                        # Try a smaller step in the original direction
                        if calls < budget and time.time() < deadline:
                            trial3 = x.copy()
                            trial3[d] += steps[d] * 0.5 * last_dirs[d]
                            trial3 = np.clip(trial3, -5.0, 5.0)
                            f_trial3 = func(trial3)
                            calls += 1
                            if f_trial3 < fx:
                                x = trial3
                                fx = f_trial3
                                improved_this_round = True

            # If no improvement in entire round, do a random perturbation restart
            if not improved_this_round:
                # Random restart with small perturbation around best
                x = best_x + rng.normal(0, 0.5, dim)
                x = np.clip(x, -5.0, 5.0)
                fx = eval_point(x)
                calls += 1
                # Reset some steps
                steps = np.ones(dim) * 1.0
                last_dirs = rng.choice([-1, 1], dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
