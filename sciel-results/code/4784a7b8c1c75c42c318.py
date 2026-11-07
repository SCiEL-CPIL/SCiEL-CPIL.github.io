"""SEARCH approach: grid-based decomposition with Nelder-Mead refinement for RE37."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using grid decomposition + Nelder-Mead."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])

        eval_count = [0]
        max_budget = self.budget

        def evaluate(x):
            eval_count[0] += 1
            if eval_count[0] > max_budget:
                raise ValueError("Budget exceeded")
            return np.array(problem(x))

        def time_ok():
            return time.time() < deadline

        def is_dominated(f, F_set):
            if len(F_set) == 0:
                return False
            for j in range(len(F_set)):
                if np.all(F_set[j] <= f) and np.any(F_set[j] < f):
                    return True
            return False

        def remove_dominated_with_indices(F_set, X_set):
            """Remove dominated points, return filtered F and X."""
            n = len(F_set)
            if n <= 1:
                return F_set, X_set
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                if not keep[i]:
                    continue
                for j in range(n):
                    if i == j or not keep[j]:
                        continue
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        keep[i] = False
                        break
            return F_set[keep], X_set[keep]

        def nelder_mead(f_obj, x0, max_iter=30, tol=1e-6):
            """Nelder-Mead minimization."""
            n = len(x0)
            simplex = [x0.copy() for _ in range(n + 1)]
            for i in range(n):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += 0.1 * (hi[i] - lo[i])

            fvals = [f_obj(s) for s in simplex]

            for it in range(max_iter):
                if not time_ok() or eval_count[0] >= max_budget:
                    break
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if abs(fvals[-1] - fvals[0]) < tol:
                    break

                centroid = np.mean(simplex[:n], axis=0)
                x_r = centroid + (centroid - simplex[-1])
                x_r = np.clip(x_r, lo, hi)
                f_r = f_obj(x_r)

                if f_r < fvals[0]:
                    x_e = centroid + 2 * (centroid - simplex[-1])
                    x_e = np.clip(x_e, lo, hi)
                    f_e = f_obj(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    x_c = centroid + 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = f_obj(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = 0.5 * (simplex[0] + simplex[i])
                            fvals[i] = f_obj(simplex[i])

            order = np.argmin(fvals)
            return simplex[order], fvals[order]

        # Phase 1: Grid-based coarse evaluation
        grid_size = 3
        grid_points = []
        for i0 in range(grid_size):
            for i1 in range(grid_size):
                for i2 in range(grid_size):
                    for i3 in range(grid_size):
                        x = np.array([
                            lo[0] + (hi[0] - lo[0]) * i0 / (grid_size - 1),
                            lo[1] + (hi[1] - lo[1]) * i1 / (grid_size - 1),
                            lo[2] + (hi[2] - lo[2]) * i2 / (grid_size - 1),
                            lo[3] + (hi[3] - lo[3]) * i3 / (grid_size - 1),
                        ])
                        grid_points.append(x)

        grid_F = []
        for x in grid_points:
            if eval_count[0] >= max_budget or not time_ok():
                break
            f = evaluate(x)
            grid_F.append(f)

        grid_F = np.array(grid_F)
        grid_points = np.array(grid_points[:len(grid_F)])

        # Phase 2: Add LHS samples for better coverage
        n_lhs = min(40, max_budget - eval_count[0])
        if n_lhs > 0 and time_ok():
            lhs_X = np.zeros((n_lhs, dim))
            for d in range(dim):
                perm = np.random.permutation(n_lhs)
                lhs_X[:, d] = (perm + np.random.random(n_lhs)) / n_lhs
            lhs_X = lo + (hi - lo) * lhs_X

            lhs_F = []
            for x in lhs_X:
                if eval_count[0] >= max_budget or not time_ok():
                    break
                f = evaluate(x)
                lhs_F.append(f)

            if lhs_F:
                all_X = np.vstack([grid_points, np.array(lhs_F)])
                all_F = np.vstack([grid_F, np.array(lhs_F)])
            else:
                all_X = grid_points
                all_F = grid_F
        else:
            all_X = grid_points
            all_F = grid_F

        # Remove dominated from initial set (with proper index tracking)
        all_F, all_X = remove_dominated_with_indices(all_F, all_X)

        # Phase 3: Refine with Nelder-Mead on weighted sums
        # Generate diverse weight vectors
        weight_vectors = []
        # Axis-aligned weights (extreme points)
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weight_vectors.append(w)
        # Equal weights
        weight_vectors.append(np.ones(n_obj) / n_obj)
        # Grid of weights (finer)
        for a in range(5):
            for b in range(5 - a):
                c = 4 - a - b
                w = np.array([a, b, c], dtype=float)
                if w.sum() > 0:
                    weight_vectors.append(w / w.sum())
        # Random weights for diversity
        for _ in range(15):
            w = np.random.random(n_obj)
            w /= w.sum()
            weight_vectors.append(w)

        # Deduplicate weight vectors (approximate)
        unique_w = []
        for w in weight_vectors:
            is_dup = False
            for uw in unique_w:
                if np.allclose(w, uw, atol=0.01):
                    is_dup = True
                    break
            if not is_dup:
                unique_w.append(w)
        weight_vectors = unique_w

        # For each weight vector, refine with Nelder-Mead
        remaining = max_budget - eval_count[0]
        n_refinements = min(len(weight_vectors), remaining // 12)

        for wi in range(n_refinements):
            if eval_count[0] >= max_budget or not time_ok():
                break

            w = weight_vectors[wi]

            # Find best point in current set for this weight
            scalar_vals = all_F @ w
            best_idx = np.argmin(scalar_vals)
            x_start = all_X[best_idx].copy()

            # Nelder-Mead on weighted sum
            def weighted_obj(x, _w=w):
                if eval_count[0] >= max_budget:
                    return np.inf
                f = evaluate(x)
                return f @ _w

            x_opt, f_opt_val = nelder_mead(weighted_obj, x_start, max_iter=25, tol=1e-5)

            # Evaluate the final point to get its objective vector
            if eval_count[0] < max_budget and time_ok():
                f_opt = evaluate(x_opt)

                if not is_dominated(f_opt, all_F):
                    all_X = np.vstack([all_X, x_opt.reshape(1, -1)])
                    all_F = np.vstack([all_F, f_opt.reshape(1, -1)])

        # Phase 4: Final polish - refine non-dominated points with local search
        all_F, all_X = remove_dominated_with_indices(all_F, all_X)
        remaining = max_budget - eval_count[0]

        if remaining > 5 and time_ok():
            n_nd = len(all_F)
            n_polish = min(n_nd, remaining // 8)

            for i in range(n_polish):
                if eval_count[0] >= max_budget or not time_ok():
                    break

                x_base = all_X[i].copy()
                f_base = all_F[i].copy()

                # Small random perturbations with decreasing scale
                for trial in range(7):
                    if eval_count[0] >= max_budget or not time_ok():
                        break
                    scale = 0.04 * (0.65 ** trial) * (hi - lo)
                    x_try = x_base + scale * np.random.standard_normal(dim)
                    x_try = np.clip(x_try, lo, hi)
                    f_try = evaluate(x_try)

                    if not is_dominated(f_try, all_F):
                        if np.all(f_try <= f_base) and np.any(f_try < f_base):
                            all_F[i] = f_try
                            all_X[i] = x_try
                            x_base = x_try
                            f_base = f_try
                        else:
                            all_X = np.vstack([all_X, x_try.reshape(1, -1)])
                            all_F = np.vstack([all_F, f_try.reshape(1, -1)])

        # Final cleanup
        all_F, all_X = remove_dominated_with_indices(all_F, all_X)

        # Return the final set of points
        if len(all_F) > 0:
            return all_X
        else:
            x = lo + 0.5 * (hi - lo)
            f = evaluate(x)
            return np.array([x])
# EVOLVE-BLOCK-END
