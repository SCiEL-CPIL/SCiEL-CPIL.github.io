"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_made = 0

        def time_left():
            return time.time() < deadline

        # Multi-start Nelder-Mead
        def nelder_mead(x0, max_iter, sigma_init=0.5):
            nonlocal calls_made
            n = dim
            # Build initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += sigma_init

            # Evaluate initial simplex
            vals = np.zeros(n + 1)
            for i in range(n + 1):
                x_i = np.clip(simplex[i], -5, 5)
                vals[i] = func(x_i)
                calls_made += 1
                if not time_left():
                    break

            best_idx = np.argmin(vals)
            best_x = simplex[best_idx].copy()
            best_f = vals[best_idx]

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for it in range(max_iter):
                if not time_left():
                    break
                if calls_made >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(vals)
                simplex = simplex[order]
                vals = vals[order]

                # Check convergence
                if np.max(np.abs(simplex - simplex[0])) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_made += 1

                if vals[0] <= f_ref < vals[-2]:
                    simplex[-1] = x_ref
                    vals[-1] = f_ref
                elif f_ref < vals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_made += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        vals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < vals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_made += 1

                    if f_con <= min(f_ref, vals[-1]):
                        simplex[-1] = x_con
                        vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            vals[i] = func(simplex[i])
                            calls_made += 1
                            if not time_left():
                                break
                        if not time_left():
                            break

                # Update best
                best_idx = np.argmin(vals)
                if vals[best_idx] < best_f:
                    best_f = vals[best_idx]
                    best_x = simplex[best_idx].copy()

            return best_x, best_f

        # Generate diverse starting points
        n_starts = min(20, budget // 10)
        start_points = []
        # Include origin and some structured points
        start_points.append(np.zeros(dim))
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4, 4, dim))

        global_best_x = start_points[0].copy()
        global_best_f = np.inf

        for sp in start_points:
            if not time_left():
                break
            if calls_made >= budget:
                break

            # Allocate remaining budget proportionally
            remaining_budget = budget - calls_made
            remaining_starts = n_starts - len(start_points) + 1
            iter_per_start = max(5, remaining_budget // max(1, remaining_starts))

            bx, bf = nelder_mead(sp, iter_per_start)
            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()

            # If we found a good solution, do a local refinement
            if time_left() and calls_made < budget and bf < global_best_f + 1.0:
                # Refine from best with smaller step
                remaining = budget - calls_made
                if remaining > 20:
                    bx2, bf2 = nelder_mead(bx, min(remaining, 50), sigma_init=0.1)
                    if bf2 < global_best_f:
                        global_best_f = bf2
                        global_best_x = bx2.copy()

        # Final refinement from global best if time/budget allows
        if time_left() and calls_made < budget:
            remaining = budget - calls_made
            if remaining > 10:
                bx, bf = nelder_mead(global_best_x, min(remaining, 100), sigma_init=0.05)
                if bf < global_best_f:
                    global_best_f = bf
                    global_best_x = bx.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
