"""Combined optimizer for BBOB f13 (Sharp ridge): deterministic coordinate descent with adaptive steps."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-start with adaptive coordinate descent for Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        box_low = -5.0
        box_high = 5.0
        calls_made = 0

        def clip(x):
            return np.clip(x, box_low, box_high)

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = clip(x)
            f = func(x)
            calls_made += 1
            return x, f

        # Halton sequence for low-discrepancy initialization
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(index, dim, low, high):
            point = np.zeros(dim)
            for d in range(dim):
                base = d + 2  # 2, 3, 5, 7, 11, ...
                point[d] = low + (high - low) * halton(index + 1, base)
            return point

        # Phase 1: Broad exploration with Halton sequence
        n_explore = min(60, budget // 8)
        best_x = None
        best_f = np.inf
        all_points = []  # store (x, f) pairs for later refinement

        for i in range(n_explore):
            if time.time() > deadline - 60:
                break
            x = halton_point(i, dim, box_low, box_high)
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            all_points.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Also evaluate origin and a few structured points
        structured = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.array([1, -1, 1, -1, 1]) * 2.0,
            np.array([3, -3, 0, 3, -3]),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
            np.array([-0.5, -0.5, -0.5, -0.5, -0.5]),
        ]
        for x in structured:
            if time.time() > deadline - 60:
                break
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            all_points.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Sort all points by function value
        all_points.sort(key=lambda t: t[1])

        # Phase 2: Adaptive coordinate descent from multiple starting points
        # This exploits the separable structure of Sharp Ridge by optimizing
        # each dimension independently with adaptive step sizes

        def adaptive_coordinate_descent(x0, f0, max_iter=100):
            """Adaptive coordinate descent with moving average of successful steps."""
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline - 30:
                return x0, f0

            current_x = x0.copy()
            current_f = f0

            # Adaptive step sizes per dimension
            steps = np.ones(dim) * 1.0
            # Moving averages of successful step magnitudes
            step_averages = np.ones(dim) * 1.0
            # Track which direction (sign) worked for each dimension
            step_directions = np.ones(dim)

            for iteration in range(max_iter):
                if time.time() > deadline - 30 or calls_made >= budget:
                    break

                improved = False
                # Cycle through dimensions in a deterministic order
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline - 30:
                        break

                    # Try current step direction first
                    trial = current_x.copy()
                    trial[d] += step_directions[d] * steps[d]
                    trial = clip(trial)
                    res = evaluate(trial)
                    if res is None:
                        break
                    trial, f_trial = res

                    if f_trial < current_f:
                        current_x = trial
                        current_f = f_trial
                        improved = True
                        # Update moving average of successful step
                        step_averages[d] = 0.7 * step_averages[d] + 0.3 * steps[d]
                        # Slightly increase step size for this dimension
                        steps[d] = min(steps[d] * 1.2, 3.0)
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = current_x.copy()
                    else:
                        # Try opposite direction
                        if calls_made < budget and time.time() < deadline - 30:
                            trial2 = current_x.copy()
                            trial2[d] -= step_directions[d] * steps[d]
                            trial2 = clip(trial2)
                            res2 = evaluate(trial2)
                            if res2 is None:
                                break
                            trial2, f_trial2 = res2

                            if f_trial2 < current_f:
                                current_x = trial2
                                current_f = f_trial2
                                improved = True
                                step_directions[d] *= -1
                                # Update moving average
                                step_averages[d] = 0.7 * step_averages[d] + 0.3 * steps[d]
                                steps[d] = min(steps[d] * 1.2, 3.0)
                                if f_trial2 < best_f:
                                    best_f = f_trial2
                                    best_x = current_x.copy()
                            else:
                                # No improvement in either direction; shrink step
                                steps[d] *= 0.7
                                # If step becomes very small, reset to average
                                if steps[d] < 0.01:
                                    steps[d] = step_averages[d] * 0.5

                if improved:
                    # Continue searching
                    pass
                else:
                    # No improvement in full cycle
                    # Check if all steps are very small (converged)
                    if np.all(steps < 0.005):
                        break
                    # Shrink all steps slightly
                    steps *= 0.9
                    # If we've been stagnant for a while, break
                    if iteration > 30 and np.all(steps < 0.01):
                        break

            return current_x, current_f

        # Run adaptive coordinate descent from the top few starting points
        n_refine = min(5, len(all_points))
        for i in range(n_refine):
            if calls_made >= budget or time.time() > deadline - 30:
                break

            x_start, f_start = all_points[i]
            x_cd, f_cd = adaptive_coordinate_descent(x_start, f_start)
            if f_cd < best_f:
                best_f = f_cd
                best_x = x_cd

        # Phase 3: Final fine-tuning with very small steps from the best point
        if calls_made < budget and time.time() < deadline - 15 and best_x is not None:
            current_x = best_x.copy()
            current_f = best_f
            fine_steps = np.ones(dim) * 0.1

            for _ in range(50):
                if calls_made >= budget or time.time() > deadline - 15:
                    break

                improved = False
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline - 15:
                        break

                    # Try positive
                    trial = current_x.copy()
                    trial[d] += fine_steps[d]
                    trial = clip(trial)
                    res = evaluate(trial)
                    if res is None:
                        break
                    trial, f_trial = res
                    if f_trial < current_f:
                        current_x = trial
                        current_f = f_trial
                        improved = True
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = current_x.copy()

                    if calls_made < budget and time.time() < deadline - 15:
                        # Try negative
                        trial = current_x.copy()
                        trial[d] -= fine_steps[d]
                        trial = clip(trial)
                        res = evaluate(trial)
                        if res is None:
                            break
                        trial, f_trial = res
                        if f_trial < current_f:
                            current_x = trial
                            current_f = f_trial
                            improved = True
                            if f_trial < best_f:
                                best_f = f_trial
                                best_x = current_x.copy()

                if improved:
                    fine_steps *= 1.1
                else:
                    fine_steps *= 0.6
                    if np.all(fine_steps < 0.001):
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
