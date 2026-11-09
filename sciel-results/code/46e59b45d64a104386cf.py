"""Optimizer for BBOB f13 (Sharp Ridge) with ridge detection and targeted search."""
import numpy as np
import time


class Optimizer:
    """Ridge-aware optimizer for BBOB f13 Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        box_low = -5.0
        box_high = 5.0

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def eval_point(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), box_low, box_high)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_left():
            return deadline - time.perf_counter()

        # Phase 1: Ridge direction detection
        # The ridge is a direction where the function is nearly flat.
        # Probe from the origin with several step sizes and directions.
        # We'll use a set of candidate directions and measure flatness.

        # Generate a set of probe directions (deterministic)
        rng = np.random.default_rng(12345)
        n_probe_dirs = 40
        probe_dirs = []
        # Include axis directions
        for d in range(dim):
            v = np.zeros(dim)
            v[d] = 1.0
            probe_dirs.append(v)
        # Add random directions
        for _ in range(n_probe_dirs - dim):
            v = rng.standard_normal(dim)
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v /= norm
            probe_dirs.append(v)

        # For each direction, measure the "flatness" at a few base points
        # Flatness: small change in f when moving along the direction
        # Use multiple base points to be robust
        base_points = [np.zeros(dim)]
        # Add a few offset base points
        for _ in range(4):
            bp = rng.uniform(-3, 3, size=dim)
            base_points.append(bp)

        # For each direction, compute average absolute change from base points
        dir_scores = np.zeros(len(probe_dirs))
        step_sizes = [0.5, 1.0, 2.0]

        for di, dvec in enumerate(probe_dirs):
            total_change = 0.0
            n_evals_for_dir = 0
            for bp in base_points:
                for step in step_sizes:
                    if calls_made >= budget - 200:
                        break
                    x_plus = bp + step * dvec
                    x_minus = bp - step * dvec
                    f_plus = eval_point(x_plus)
                    f_minus = eval_point(x_minus)
                    # Change relative to midpoint
                    # Actually just use |f_plus - f_minus| as a measure of slope
                    # But we want to normalize. The ridge direction has small |f_plus - f_minus|
                    total_change += abs(f_plus - f_minus)
                    n_evals_for_dir += 1
                if calls_made >= budget - 200:
                    break
            if n_evals_for_dir > 0:
                dir_scores[di] = total_change / n_evals_for_dir
            else:
                dir_scores[di] = np.inf

        # The ridge direction is the one with smallest score (flattest)
        # But we also need to check: maybe all directions are similar (no clear ridge)
        # Sort by score
        sorted_indices = np.argsort(dir_scores)
        ridge_dir = probe_dirs[sorted_indices[0]].copy()
        ridge_dir = np.array(ridge_dir, dtype=float)
        ridge_norm = np.linalg.norm(ridge_dir)
        if ridge_norm > 1e-10:
            ridge_dir /= ridge_norm
        else:
            ridge_dir = np.zeros(dim)
            ridge_dir[0] = 1.0

        # Phase 2: Search along the ridge and perpendicular
        # First, do a 1D search along the ridge direction from the best point
        # Then do coordinate descent in the perpendicular subspace

        # Build an orthonormal basis where the first vector is the ridge direction
        basis = np.zeros((dim, dim))
        basis[0] = ridge_dir
        # Complete the basis using Gram-Schmidt
        for i in range(1, dim):
            v = rng.standard_normal(dim)
            # Subtract projections onto existing basis vectors
            for j in range(i):
                v = v - np.dot(v, basis[j]) * basis[j]
            norm = np.linalg.norm(v)
            if norm < 1e-10:
                # Fallback: use a standard basis vector
                v = np.zeros(dim)
                v[i] = 1.0
                for j in range(i):
                    v = v - np.dot(v, basis[j]) * basis[j]
                norm = np.linalg.norm(v)
            if norm > 1e-10:
                basis[i] = v / norm
            else:
                basis[i] = np.zeros(dim)
                basis[i][i] = 1.0

        # 1D search along ridge from best point
        # Use golden section search
        def golden_search_1d(start_coord, direction, f_at_best, lo, hi):
            """Golden section search along a direction."""
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            inv_phi = 1.0 / phi
            inv_phi2 = 1.0 / (phi * phi)

            a = lo
            b = hi
            # Initial points
            c = b - inv_phi * (b - a)
            d = a + inv_phi * (b - a)

            base = best_x.copy()
            # Evaluate at c and d
            x_c = base + c * direction
            x_d = base + d * direction
            f_c = eval_point(x_c)
            f_d = eval_point(x_d)

            for _ in range(30):
                if calls_made >= budget - 100:
                    break
                if time_left() < 20:
                    break
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - inv_phi * (b - a)
                    x_c = base + c * direction
                    f_c = eval_point(x_c)
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + inv_phi * (b - a)
                    x_d = base + d * direction
                    f_d = eval_point(x_d)

            # Return the better of the last two
            if f_c < f_d:
                return c, f_c
            else:
                return d, f_d

        # Do golden section search along the ridge from the best point
        # The ridge coordinate of best_x
        best_coord_along_ridge = np.dot(best_x, ridge_dir)
        # Search in a wide range
        lo_coord = -5.0
        hi_coord = 5.0
        # Actually, the ridge might extend beyond the box. Let's search in a reasonable range.
        # The minimizer is in [-4, 4]^5, so the ridge coordinate is in roughly [-4*sqrt(5), 4*sqrt(5)]
        # But let's be conservative: [-7, 7]
        lo_coord = -7.0
        hi_coord = 7.0

        ridge_coord, ridge_f = golden_search_1d(best_coord_along_ridge, ridge_dir, best_f, lo_coord, hi_coord)
        new_x = best_x + (ridge_coord - best_coord_along_ridge) * ridge_dir
        new_x = np.clip(new_x, box_low, box_high)
        f_new = eval_point(new_x)

        # Now do coordinate descent in the perpendicular subspace
        # For each perpendicular direction, do a 1D search
        for i in range(1, dim):
            if calls_made >= budget - 50:
                break
            if time_left() < 15:
                break
            dvec = basis[i]
            # Current coordinate along this direction
            coord = np.dot(best_x, dvec)
            # Search in a small range around the current coordinate
            lo_c = coord - 3.0
            hi_c = coord + 3.0
            c_opt, f_opt_c = golden_search_1d(coord, dvec, best_f, lo_c, hi_c)
            # Update best_x
            best_x = best_x + (c_opt - coord) * dvec
            best_x = np.clip(best_x, box_low, box_high)
            f_update = eval_point(best_x)

        # Phase 3: Fine local refinement using (1+1)-ES
        # This helps fine-tune the solution
        if calls_made < budget - 20 and time_left() > 10:
            # (1+1)-ES with adaptive step size
            sigma = 0.1
            current_x = best_x.copy()
            current_f = best_f
            n_iter = 0
            max_iter = min(budget - calls_made - 10, 200)

            while n_iter < max_iter and calls_made < budget - 10 and time_left() > 5:
                # Generate a mutation
                mutation = rng.standard_normal(dim) * sigma
                trial_x = current_x + mutation
                trial_x = np.clip(trial_x, box_low, box_high)
                trial_f = eval_point(trial_x)

                if trial_f < current_f:
                    current_x = trial_x
                    current_f = trial_f
                    sigma = min(sigma * 1.2, 1.0)
                else:
                    sigma = max(sigma * 0.7, 0.001)

                n_iter += 1

        # Phase 4: Final Nelder-Mead refinement if time permits
        if calls_made < budget - 30 and time_left() > 10:
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead for final refinement
                x0 = best_x.copy()
                res = minimize(
                    lambda x: func(np.clip(x, box_low, box_high)),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': min(budget - calls_made - 5, 100),
                             'xatol': 1e-6,
                             'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_x = np.clip(res.x, box_low, box_high)
                    best_f = res.fun
            except Exception:
                pass

        return best_x, best_f
