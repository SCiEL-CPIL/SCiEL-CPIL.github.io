"""Multi-objective optimizer for WFG4 using LHS-based decomposition with
coordinate descent refinement and final dense sweep."""
import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        mid = 0.5 * (lo + hi)
        budget = self.budget

        eval_count = 0
        archive_X = []
        archive_F = []

        def time_left():
            return deadline - time.time()

        def evaluate(x):
            nonlocal eval_count
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            eval_count += 1
            archive_X.append(x.copy())
            archive_F.append(f.copy())
            return f

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def is_dominating(f, archive_F):
            """Check if f dominates any point in archive_F."""
            for i, af in enumerate(archive_F):
                if np.all(f <= af) and np.any(f < af):
                    return i
            return -1

        def update_archive(x, f):
            """Add to archive if not dominated, remove dominated points."""
            if is_dominated(f, archive_F):
                return False
            # Remove points dominated by f
            to_remove = []
            for i, af in enumerate(archive_F):
                if np.all(f <= af) and np.any(f < af):
                    to_remove.append(i)
            for i in reversed(to_remove):
                archive_F.pop(i)
                archive_X.pop(i)
            archive_X.append(x.copy())
            archive_F.append(f.copy())
            return True

        def coordinate_descent(x, f, max_iters, step_frac=0.1):
            """Perform coordinate descent to minimize sum of objectives."""
            for iteration in range(max_iters):
                if eval_count >= budget or time_left() < 2.0:
                    break
                improved = False
                for i in range(dim):
                    if eval_count >= budget or time_left() < 1.0:
                        break
                    best_sum = np.sum(f)
                    best_val = x[i]
                    best_f = f
                    # Try several trial values around current
                    trial_vals = [
                        lo[i], hi[i],
                        x[i] - step_frac * span[i],
                        x[i] + step_frac * span[i],
                        0.25 * lo[i] + 0.75 * hi[i],
                        0.5 * lo[i] + 0.5 * hi[i],
                        0.75 * lo[i] + 0.25 * hi[i],
                    ]
                    for tv in trial_vals:
                        tv = np.clip(tv, lo[i], hi[i])
                        if abs(tv - x[i]) < 1e-10:
                            continue
                        if eval_count >= budget or time_left() < 1.0:
                            break
                        x_trial = x.copy()
                        x_trial[i] = tv
                        f_trial = evaluate(x_trial)
                        if np.sum(f_trial) < best_sum:
                            best_sum = np.sum(f_trial)
                            best_val = tv
                            best_f = f_trial
                    if abs(best_val - x[i]) > 1e-10:
                        x[i] = best_val
                        f = best_f
                        improved = True
                if not improved:
                    break
            return x, f

        # ---- Phase 1: Latin Hypercube Sampling for initial coverage ----
        # Generate LHS points: one per cell in a grid
        n_lhs = min(30, budget // 5)  # Use 30 points for LHS
        # Create LHS: for each dimension, permute the intervals
        lhs_points = []
        for _ in range(n_lhs):
            x = np.zeros(dim)
            for d in range(dim):
                # Random permutation for this dimension
                perm = np.random.permutation(n_lhs)
                # Assign the current point to a random interval
                idx = np.random.randint(n_lhs)
                interval = perm[idx]
                # Random position within the interval
                frac = (interval + np.random.random()) / n_lhs
                x[d] = lo[d] + frac * span[d]
            lhs_points.append(x)

        for x in lhs_points:
            if eval_count >= budget or time_left() < 5.0:
                break
            f = evaluate(x)
            update_archive(x, f)

        # ---- Phase 2: Identify front variable and do initial sweep ----
        # Determine which variable has the most impact on objectives
        if len(archive_F) >= 2:
            F_arr = np.array(archive_F)
            # Estimate impact of each variable by looking at variance
            impact = np.zeros(dim)
            for d in range(dim):
                # Correlation of x[d] with each objective
                X_col = np.array([x[d] for x in archive_X])
                for k in range(n_obj):
                    F_col = F_arr[:, k]
                    if np.std(X_col) > 1e-10 and np.std(F_col) > 1e-10:
                        corr = np.corrcoef(X_col, F_col)[0, 1]
                        impact[d] += abs(corr)
            front_var = int(np.argmax(impact))
        else:
            front_var = 0

        # Coarse sweep along front variable
        n_coarse = 12
        t_vals = np.linspace(0.0, 1.0, n_coarse)
        for t in t_vals:
            if eval_count >= budget or time_left() < 5.0:
                break
            x = mid.copy()
            x[front_var] = lo[front_var] + t * span[front_var]
            f = evaluate(x)
            update_archive(x, f)

        # ---- Phase 3: Coordinate descent refinement on archive points ----
        # Refine each point in the archive
        remaining = budget - eval_count
        n_refine = min(len(archive_X), max(5, remaining // 20))
        # Pick the most promising points (lowest sum of objectives)
        if len(archive_F) > n_refine:
            sums = np.sum(np.array(archive_F), axis=1)
            indices = np.argsort(sums)[:n_refine]
        else:
            indices = list(range(len(archive_F)))

        for idx in indices:
            if eval_count >= budget or time_left() < 5.0:
                break
            x = archive_X[idx].copy()
            f = archive_F[idx].copy()
            x, f = coordinate_descent(x, f, max_iters=3)
            update_archive(x, f)

        # ---- Phase 4: Dense sweep along front with optimized penalties ----
        # Determine optimal penalty configuration (median of refined points)
        optimal_penalty = mid.copy()
        if archive_X:
            for i in range(dim):
                if i == front_var:
                    continue
                vals = np.array([x[i] for x in archive_X])
                optimal_penalty[i] = np.median(vals)

        remaining = budget - eval_count
        n_dense = max(10, min(80, int(remaining * 0.7)))
        n_dense = max(5, min(n_dense, remaining))
        dense_ts = np.linspace(0.0, 1.0, n_dense)

        for t in dense_ts:
            if eval_count >= budget or time_left() < 1.0:
                break
            x = optimal_penalty.copy()
            x[front_var] = lo[front_var] + t * span[front_var]
            f = evaluate(x)
            update_archive(x, f)

        # ---- Phase 5: Local refinement around best points ----
        remaining = budget - eval_count
        if remaining > 15 and time_left() > 10.0:
            # Pick top points by sum of objectives
            if archive_F:
                F_arr = np.array(archive_F)
                sums = np.sum(F_arr, axis=1)
                top_indices = np.argsort(sums)[:min(5, len(archive_F))]

                for idx in top_indices:
                    if eval_count >= budget or time_left() < 5.0:
                        break
                    best_x = archive_X[idx].copy()
                    best_f = archive_F[idx].copy()

                    sigma = 0.03 * np.max(span)
                    C = np.eye(dim)
                    n_iter = min(5, (budget - eval_count) // dim)

                    for _ in range(n_iter):
                        if eval_count >= budget or time_left() < 2.0:
                            break
                        z = np.random.standard_normal(dim)
                        x_new = best_x + sigma * z
                        x_new = np.clip(x_new, lo, hi)
                        f_new = evaluate(x_new)

                        if np.sum(f_new) < np.sum(best_f):
                            diff = x_new - best_x
                            C = 0.7 * C + 0.3 * (np.outer(diff, diff) / (sigma**2))
                            C = (C + C.T) / 2
                            C += 1e-6 * np.eye(dim)
                            best_x = x_new
                            best_f = f_new
                            sigma *= 0.9
                            update_archive(best_x, best_f)
                        else:
                            sigma *= 0.85

        return np.array(archive_X) if archive_X else np.array([mid])
