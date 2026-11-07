"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hooke-Jeeves pattern search with adaptive steps and quadratic refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Count evaluations
        self._eval_count = 0

        def eval_func(x):
            self._eval_count += 1
            return func(x)

        dim = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = eval_func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step sizes per coordinate
        steps = np.ones(dim) * 1.0
        min_step = 1e-8

        # Hooke-Jeeves pattern search
        while self._eval_count < budget:
            if time.time() > deadline:
                break

            improved = False
            for i in range(dim):
                if time.time() > deadline:
                    break

                # Try positive direction
                x_trial = x.copy()
                x_trial[i] = np.clip(x_trial[i] + steps[i], -5.0, 5.0)
                fx_trial = eval_func(x_trial)

                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    steps[i] *= 2.0
                    improved = True

                    # Continue in the same direction while improving
                    while self._eval_count < budget:
                        if time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[i] = np.clip(x_trial[i] + steps[i], -5.0, 5.0)
                        fx_trial = eval_func(x_trial)
                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                        else:
                            break
                else:
                    # Try negative direction
                    x_trial = x.copy()
                    x_trial[i] = np.clip(x_trial[i] - steps[i], -5.0, 5.0)
                    fx_trial = eval_func(x_trial)

                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        steps[i] *= 2.0
                        improved = True

                        # Continue in the same direction while improving
                        while self._eval_count < budget:
                            if time.time() > deadline:
                                break
                            x_trial = x.copy()
                            x_trial[i] = np.clip(x_trial[i] - steps[i], -5.0, 5.0)
                            fx_trial = eval_func(x_trial)
                            if fx_trial < fx:
                                x = x_trial
                                fx = fx_trial
                            else:
                                break
                    else:
                        steps[i] *= 0.5

                if steps[i] < min_step:
                    steps[i] = min_step

            if not improved:
                # No improvement in full cycle; shrink all steps
                steps *= 0.5
                if np.max(steps) < min_step:
                    break

            # Update best
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

        # Quadratic refinement using recent points if budget allows
        if self._eval_count < budget and time.time() < deadline:
            # Collect a small set of points around best_x for quadratic fit
            # Use coordinate-wise 1D quadratic fits
            local_step = 0.1
            for i in range(dim):
                if self._eval_count >= budget - 1:
                    break
                if time.time() > deadline:
                    break

                # Evaluate at best_x +/- local_step in coordinate i
                x_minus = best_x.copy()
                x_plus = best_x.copy()
                x_minus[i] = np.clip(x_minus[i] - local_step, -5.0, 5.0)
                x_plus[i] = np.clip(x_plus[i] + local_step, -5.0, 5.0)

                f_minus = eval_func(x_minus)
                if self._eval_count >= budget - 1:
                    break
                f_plus = eval_func(x_plus)

                # Quadratic fit: f(t) = a*t^2 + b*t + c where t is offset from best_x[i]
                # Using points at -local_step, 0, +local_step
                t1, t2, t3 = -local_step, 0.0, local_step
                f1, f2, f3 = f_minus, best_fx, f_plus

                # Solve for vertex: t* = -b/(2a)
                # Using Lagrange or direct formula
                denom = (t1 - t2) * (t1 - t3) * (t2 - t3)
                if abs(denom) < 1e-30:
                    continue
                a = (f1 / ((t1 - t2) * (t1 - t3)) +
                     f2 / ((t2 - t1) * (t2 - t3)) +
                     f3 / ((t3 - t1) * (t3 - t2)))
                b = -(f1 * (t1 + t2 + t3) / ((t1 - t2) * (t1 - t3)) +
                      f2 * (t1 + t2 + t3) / ((t2 - t1) * (t2 - t3)) +
                      f3 * (t1 + t2 + t3) / ((t3 - t1) * (t3 - t2)))

                if abs(a) > 1e-15:
                    t_star = -b / (2.0 * a)
                    if abs(t_star) > 1e-10:
                        x_new = best_x.copy()
                        x_new[i] = np.clip(x_new[i] + t_star, -5.0, 5.0)
                        f_new = eval_func(x_new)
                        if f_new < best_fx:
                            best_x = x_new
                            best_fx = f_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
