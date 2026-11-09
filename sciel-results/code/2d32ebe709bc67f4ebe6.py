"""Rewritten optimizer: exploit Rastrigin separability via per-coordinate 1D grids + Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Per-coordinate 1D grid search to find the Rastrigin basin, then Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def try_point(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Per-coordinate 1D grid search to locate the global basin ---
        # Rastrigin f(x) = A*d + sum(x_i^2 - A*cos(2*pi*x_i))
        # The cosine term has period 2 and minima at even integers.
        # The x_i^2 term biases toward x_i=0.
        # With hidden shift s_i, the minimum is near s_i (if |s_i| is small) or
        # near the nearest even integer to s_i (if the quadratic penalty is small).
        #
        # Strategy: for each coordinate, evaluate a fine 1D grid while keeping
        # other coordinates at 0. This identifies which coordinate values give
        # low function values, revealing the shift pattern.

        # Number of grid points per coordinate
        n_grid = 41  # spacing = 10/40 = 0.25
        grid_vals = np.linspace(-4.0, 4.0, n_grid)  # minimizer is in [-4,4]^5

        # For each dimension, sweep that coordinate while others are at 0
        # This costs dim * n_grid = 5 * 41 = 205 calls
        coord_best = np.zeros(dim)
        coord_best_f = np.zeros(dim)

        for d in range(dim):
            local_best_val = 0.0
            local_best_f = float('inf')
            for gv in grid_vals:
                if not time_ok() or calls_made >= budget:
                    break
                x = np.zeros(dim)
                x[d] = gv
                fx = try_point(x)
                if fx is not None and fx < local_best_f:
                    local_best_f = fx
                    local_best_val = gv
            coord_best[d] = local_best_val

        # Build initial candidate from per-coordinate bests
        x0 = coord_best.copy()
        try_point(x0)

        # --- Phase 2: Joint refinement ---
        # Now that we have a good starting point, do coordinate-wise refinement
        # with finer grids around the identified basin.

        for pass_num in range(4):
            if not time_ok() or calls_made >= budget:
                break
            for d in range(dim):
                if not time_ok() or calls_made >= budget:
                    break
                # Fine grid around current best for this coordinate
                center = best_x[d]
                # Adaptive range: start wide, narrow down
                if pass_num == 0:
                    half_range = 1.0
                    n_pts = 21
                elif pass_num == 1:
                    half_range = 0.5
                    n_pts = 21
                elif pass_num == 2:
                    half_range = 0.25
                    n_pts = 21
                else:
                    half_range = 0.125
                    n_pts = 21

                a = max(-5.0, center - half_range)
                b = min(5.0, center + half_range)
                fine_grid = np.linspace(a, b, n_pts)

                local_best_val = center
                local_best_f = best_f
                for gv in fine_grid:
                    if not time_ok() or calls_made >= budget:
                        break
                    x_try = best_x.copy()
                    x_try[d] = gv
                    fx = try_point(x_try)
                    if fx is not None and fx < local_best_f:
                        local_best_f = fx
                        local_best_val = gv

                best_x[d] = local_best_val

        # --- Phase 3: Nelder-Mead polish ---
        if best_x is not None and time_ok() and calls_made < budget:
            def nm_minimize(func, x0, budget_left, deadline):
                dim = len(x0)
                simplex = [x0.copy()]
                f0 = func(np.clip(x0, -5, 5))
                simplex_f = [f0]
                calls = 1

                for i in range(dim):
                    if calls >= budget_left or time.time() >= deadline:
                        break
                    x_i = x0.copy()
                    x_i[i] += 0.05
                    simplex.append(x_i)
                    f_i = func(np.clip(x_i, -5, 5))
                    simplex_f.append(f_i)
                    calls += 1

                if len(simplex) < 2:
                    return x0, f0, calls

                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5

                while calls < budget_left and time.time() < deadline:
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    centroid = np.mean(simplex[:-1], axis=0)

                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    calls += 1
                    if calls >= budget_left or time.time() >= deadline:
                        break

                    if f_r < simplex_f[0]:
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = func(x_e)
                        calls += 1
                        if calls >= budget_left or time.time() >= deadline:
                            break
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        if f_r < simplex_f[-1]:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls += 1
                        if calls >= budget_left or time.time() >= deadline:
                            break
                        if f_c < min(f_r, simplex_f[-1]):
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            for i in range(1, dim + 1):
                                if calls >= budget_left or time.time() >= deadline:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = func(np.clip(simplex[i], -5, 5))
                                calls += 1
                            if calls >= budget_left or time.time() >= deadline:
                                break

                    min_idx = np.argmin(simplex_f)
                    if simplex_f[min_idx] < best_f:
                        best_f = simplex_f[min_idx]
                        best_x = simplex[min_idx].copy()

                    # Convergence check
                    if abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                        break

                min_idx = np.argmin(simplex_f)
                return simplex[min_idx], simplex_f[min_idx], calls

            nm_budget = budget - calls_made
            if nm_budget > dim + 1:
                x_nm, f_nm, nm_calls = nm_minimize(func, best_x, nm_budget, deadline)
                calls_made += nm_calls
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
