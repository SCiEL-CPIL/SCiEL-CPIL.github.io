"""Combined optimizer: Nelder-Mead with adaptive restarts and (1+1)-ES polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with multi-start, adaptive restarts, and (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            if calls_made[0] >= budget:
                raise Exception("Budget exceeded")

            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            calls_made[0] += 1

            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(12345)

        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma_shrink = 0.5  # Shrinkage

        # Initial simplex size
        init_sigma = 1.0

        def run_nelder_mead(start_x, max_iter, init_step=init_sigma):
            """Run Nelder-Mead from start_x for up to max_iter evaluations."""
            nonlocal calls_made, best_f, best_x, deadline

            # Initialize simplex
            simplex = [start_x.copy()]
            for i in range(n):
                pt = start_x.copy()
                step = init_step
                if start_x[i] + step > 5:
                    pt[i] = start_x[i] - step
                elif start_x[i] - step < -5:
                    pt[i] = start_x[i] + step
                else:
                    pt[i] = start_x[i] + step
                simplex.append(pt)

            # Evaluate initial simplex
            f_values = []
            for pt in simplex:
                if calls_made[0] >= budget or time.time() > deadline:
                    return
                f_values.append(wrapped_func(pt))

            if calls_made[0] >= budget:
                return

            iter_count = 0
            stall_count = 0
            prev_best = best_f

            while calls_made[0] < budget and iter_count < max_iter:
                if time.time() > deadline:
                    break

                # Sort simplex by fitness
                sorted_indices = np.argsort(f_values)
                simplex = [simplex[i] for i in sorted_indices]
                f_values = [f_values[i] for i in sorted_indices]

                # Check for convergence/stall
                if f_values[0] < prev_best - 1e-10:
                    prev_best = f_values[0]
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 25:
                        break

                # Get worst and best
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                x_best = simplex[0]

                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)

                if calls_made[0] >= budget:
                    break

                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)

                    if calls_made[0] >= budget:
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)

                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)

                    if calls_made[0] >= budget:
                        break

                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = x_best + sigma_shrink * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_made[0] >= budget or time.time() > deadline:
                                return
                            f_values[i] = wrapped_func(simplex[i])

                        if calls_made[0] >= budget:
                            break

                        # Check if simplex is too small
                        simplex_size = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, n + 1)])
                        if simplex_size < 1e-6:
                            break

                iter_count += 1

        # Multi-start strategy with coordinate-descent warm start
        # First, do a quick coordinate descent to get a good starting point
        x_cd = np.zeros(n)
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
        
        for step in step_sizes:
            if calls_made[0] >= budget or time.time() > deadline:
                break
            improved = True
            while improved and calls_made[0] < budget:
                if time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    # Try positive step
                    x_trial = x_cd.copy()
                    x_trial[i] += step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = wrapped_func(x_trial)
                    if f_trial < best_f:
                        x_cd = x_trial
                        improved = True
                    # Try negative step
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    x_trial = x_cd.copy()
                    x_trial[i] -= step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = wrapped_func(x_trial)
                    if f_trial < best_f:
                        x_cd = x_trial
                        improved = True

        # Start Nelder-Mead from the coordinate descent result
        run_nelder_mead(x_cd, max_iter=150, init_step=1.0)

        # If we still have budget, restart from best with perturbation
        while calls_made[0] < budget:
            if time.time() > deadline:
                break

            # Perturb best point with decreasing step sizes
            perturb_scale = 0.3 * (calls_made[0] / budget) + 0.5
            perturbation = rng.standard_normal(n) * perturb_scale
            new_start = best_x + perturbation
            new_start = np.clip(new_start, -5, 5)

            # Run Nelder-Mead from this new start with smaller initial step
            init_step = 0.5 * (1.0 - calls_made[0] / budget) + 0.1
            run_nelder_mead(new_start, max_iter=50, init_step=init_step)

        # Final polish with (1+1)-ES
        if calls_made[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            polish_stall = 0
            max_polish_stall = 20

            while calls_made[0] < budget:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = wrapped_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    polish_stall = 0
                else:
                    sigma *= 0.85
                    polish_stall += 1

                if polish_stall >= max_polish_stall:
                    polish_stall = 0
                    sigma = max(sigma * 0.5, 0.003)
                    # Try perturbation around best
                    for _ in range(3):
                        if time.time() > deadline or calls_made[0] >= budget:
                            break
                        y2 = np.clip(best_x + sigma * rng.standard_normal(n), -5, 5)
                        fy2 = wrapped_func(y2)
                        if fy2 < fx:
                            x, fx = y2, fy2
                            break
                        if fy2 < best_f:
                            best_f = fy2
                            best_x = y2.copy()

                    if sigma < 0.001:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
