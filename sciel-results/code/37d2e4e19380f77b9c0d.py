"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Uses (1+1)-ES with restarts and Nelder-Mead local polish.
"""
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with restarts and Nelder-Mead polish for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        def still_ok():
            return time.time() < deadline

        def nelder_mead(func, x0, f0, calls_remaining, max_iter=50):
            """Nelder-Mead local optimization."""
            if calls_remaining < 20:
                return x0, f0, 0

            # Build initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                step = 0.25 if abs(x0[i]) > 1e-4 else 0.25
                xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                fi = float(func(xi))
                calls_remaining -= 1
                simplex.append((xi, fi))

            # Sort simplex
            simplex.sort(key=lambda t: t[1])

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5  # shrink

            calls_used = dim + 1
            for _ in range(max_iter):
                if calls_remaining - calls_used < 10:
                    break

                # Centroid of all but worst
                centroid = np.zeros(dim)
                for i in range(dim):
                    centroid += simplex[i][0]
                centroid /= dim

                # Reflect worst point
                worst_idx = dim
                xr = centroid + alpha * (centroid - simplex[worst_idx][0])
                xr = np.clip(xr, -5.0, 5.0)
                fr = float(func(xr))
                calls_used += 1

                if fr < simplex[0][1]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[worst_idx][0])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = float(func(xe))
                    calls_used += 1
                    if fe < fr:
                        simplex[worst_idx] = (xe, fe)
                    else:
                        simplex[worst_idx] = (xr, fr)
                elif fr < simplex[dim - 1][1]:
                    simplex[worst_idx] = (xr, fr)
                else:
                    # Contraction
                    if fr < simplex[dim - 1][1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[worst_idx][0] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[worst_idx][0])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = float(func(xc))
                    calls_used += 1

                    if fc < simplex[dim - 1][1]:
                        simplex[worst_idx] = (xc, fc)
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i][0] = simplex[0][0] + sigma_nm * (simplex[i][0] - simplex[0][0])
                            simplex[i][0] = np.clip(simplex[i][0], -5.0, 5.0)
                            simplex[i][1] = float(func(simplex[i][0]))
                            calls_used += 1

                simplex.sort(key=lambda t: t[1])

                # Check convergence
                if abs(simplex[0][1] - simplex[dim][1]) < 1e-12:
                    break

            return simplex[0][0], simplex[0][1], calls_used

        best_x_overall = np.zeros(dim)
        best_f_overall = np.inf

        # Phase 1: Robust initialization
        n_init = 15
        init_points = [np.zeros(dim)]  # origin
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-5.0, 5.0, dim))

        init_results = []
        for x in init_points:
            f = float(func(x))
            init_results.append((x.copy(), f))
            if f < best_f_overall:
                best_f_overall = f
                best_x_overall = x.copy()

        calls_made = len(init_points)

        # Phase 2: (1+1)-ES with restarts
        # Target: use ~60% of budget for ES, ~40% for NM polish
        es_budget = int(budget * 0.6)
        max_es_calls = es_budget - calls_made

        # (1+1)-ES parameters
        target_success = 0.2  # 1/5
        sigma = 2.0
        x = best_x_overall.copy()
        f_best = best_f_overall

        no_improve_count = 0
        no_improve_limit = 50  # restart after 50 steps without improvement

        while calls_made < budget and still_ok():
            # Generate one offspring
            y = rng.standard_normal(dim)
            x_new = x + sigma * y
            x_new = np.clip(x_new, -5.0, 5.0)

            f_new = float(func(x_new))
            calls_made += 1

            if f_new < f_best:
                # Success: accept
                x = x_new
                f_best = f_new
                if f_best < best_f_overall:
                    best_f_overall = f_best
                    best_x_overall = x_new.copy()
                no_improve_count = 0
                # Increase sigma slightly on success
                sigma *= 1.1
            else:
                no_improve_count += 1
                # Decrease sigma on failure
                sigma *= 0.9

            # Keep sigma in reasonable range
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Check for restart
            if no_improve_count >= no_improve_limit:
                # Restart from a random point near current best
                if still_ok() and calls_made < budget - 30:
                    # Try a few random restarts
                    for _ in range(3):
                        if calls_made >= budget:
                            break
                        x_restart = best_x_overall + rng.standard_normal(dim) * sigma
                        x_restart = np.clip(x_restart, -5.0, 5.0)
                        f_restart = float(func(x_restart))
                        calls_made += 1
                        if f_restart < f_best:
                            x = x_restart
                            f_best = f_restart
                            if f_best < best_f_overall:
                                best_f_overall = f_best
                                best_x_overall = x_restart.copy()
                            no_improve_count = 0
                            break
                    else:
                        # No improvement from restarts, reduce sigma more aggressively
                        sigma *= 0.5
                        no_improve_count = 0

            # Adaptive sigma based on success rate (simple version)
            # Every 20 steps, adjust sigma based on recent success rate
            if no_improve_count == 0:
                pass  # already adjusted

        # Phase 3: Nelder-Mead polish
        if still_ok() and calls_made < budget:
            x_nm, f_nm, calls_used = nelder_mead(
                func, best_x_overall, best_f_overall,
                budget - calls_made, max_iter=50
            )
            calls_made += calls_used
            if f_nm < best_f_overall:
                best_f_overall = f_nm
                best_x_overall = x_nm.copy()

        # Phase 4: If time/calls remain, do another round of (1+1)-ES from best
        if still_ok() and calls_made < budget:
            sigma = 0.5
            x = best_x_overall.copy()
            f_best = best_f_overall
            no_improve_count = 0
            no_improve_limit = 30

            while calls_made < budget and still_ok():
                y = rng.standard_normal(dim)
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5.0, 5.0)

                f_new = float(func(x_new))
                calls_made += 1

                if f_new < f_best:
                    x = x_new
                    f_best = f_new
                    if f_best < best_f_overall:
                        best_f_overall = f_best
                        best_x_overall = x_new.copy()
                    no_improve_count = 0
                    sigma *= 1.05
                else:
                    no_improve_count += 1
                    sigma *= 0.95

                sigma = np.clip(sigma, 1e-8, 5.0)

                if no_improve_count >= no_improve_limit:
                    break

        return best_x_overall, float(best_f_overall)
