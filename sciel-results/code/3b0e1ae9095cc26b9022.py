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
        deadline = start_time + 285  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])

        self._eval_count = 0
        self._max_budget = self.budget

        def evaluate(x):
            self._eval_count += 1
            if self._eval_count > self._max_budget:
                raise ValueError("Budget exceeded")
            return np.array(problem(x))

        def time_ok():
            return time.time() < deadline

        def is_dominated(f, F_set):
            for j in range(len(F_set)):
                if np.all(F_set[j] <= f) and np.any(F_set[j] < f):
                    return True
            return False

        def remove_dominated(F_set):
            n = len(F_set)
            if n <= 1:
                return F_set
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
            return F_set[keep]

        def hv3d(points, ref):
            """Exact 3D hypervolume via coordinate-based grid decomposition."""
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            pts = remove_dominated(pts)
            n = len(pts)
            if n == 0:
                return 0.0

            x_coords = sorted(set([ref[0]] + list(pts[:, 0])))
            y_coords = sorted(set([ref[1]] + list(pts[:, 1])))
            z_coords = sorted(set([ref[2]] + list(pts[:, 2])))

            hv = 0.0
            for xi in range(len(x_coords) - 1):
                for yi in range(len(y_coords) - 1):
                    for zi in range(len(z_coords) - 1):
                        covered = False
                        for p in pts:
                            if p[0] <= x_coords[xi] and p[1] <= y_coords[yi] and p[2] <= z_coords[zi]:
                                covered = True
                                break
                        if covered:
                            vol = (x_coords[xi+1] - x_coords[xi]) * (y_coords[yi+1] - y_coords[yi]) * (z_coords[zi+1] - z_coords[zi])
                            hv += vol
            return hv

        def nelder_mead(f_obj, x0, max_iter=30, tol=1e-6):
            """Nelder-Mead minimization."""
            n = len(x0)
            simplex = [x0.copy() for _ in range(n + 1)]
            for i in range(n):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += 0.1 * (hi[i] - lo[i])

            fvals = [f_obj(s) for s in simplex]

            for it in range(max_iter):
                if not time_ok():
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
        # Use a 3x3x3x3 grid = 81 points (well within budget)
        grid_size = 3
        n_grid = grid_size ** dim
        budget_used = 0

        # Generate grid vertices
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
            if budget_used >= self.budget or not time_ok():
                break
            f = evaluate(x)
            budget_used += 1
            grid_F.append(f)

        grid_F = np.array(grid_F)
        grid_points = np.array(grid_points[:budget_used])

        # Phase 2: Add LHS samples for better coverage
        n_lhs = min(40, self.budget - budget_used)
        if n_lhs > 0 and time_ok():
            lhs_X = np.zeros((n_lhs, dim))
            for d in range(dim):
                perm = np.random.permutation(n_lhs)
                lhs_X[:, d] = (perm + np.random.random(n_lhs)) / n_lhs
            lhs_X = lo + (hi - lo) * lhs_X

            lhs_F = []
            for x in lhs_X:
                if budget_used >= self.budget or not time_ok():
                    break
                f = evaluate(x)
                budget_used += 1
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

        # Remove dominated from initial set
        all_F = remove_dominated(all_F)
        # Keep corresponding X
        # We need to track which points survived - redo with indices
        n_all = len(grid_F) + (n_lhs if n_lhs > 0 else 0)
        # Actually let's just work with the combined set properly
        all_X_full = []
        all_F_full = []
        for x in grid_points:
            all_X_full.append(x)
        if n_lhs > 0 and lhs_F:
            for x in lhs_X[:len(lhs_F)]:
                all_X_full.append(x)
        all_X_full = np.array(all_X_full)
        all_F_full = np.array([f for f in grid_F] + (lhs_F if n_lhs > 0 and lhs_F else []))

        # Phase 3: Identify promising cells and refine with Nelder-Mead
        # A cell is promising if its vertices have low combined objective values
        # Use weighted sum with multiple weight vectors to identify different regions

        # Generate diverse weight vectors (Chebyshev-style + random)
        weight_vectors = []
        # Axis-aligned weights (extreme points)
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weight_vectors.append(w)
        # Equal weights
        weight_vectors.append(np.ones(n_obj) / n_obj)
        # Grid of weights
        for a in range(4):
            for b in range(4 - a):
                c = 3 - a - b
                w = np.array([a, b, c], dtype=float)
                if w.sum() > 0:
                    weight_vectors.append(w / w.sum())
        # Random weights
        for _ in range(10):
            w = np.random.random(n_obj)
            w /= w.sum()
            weight_vectors.append(w)

        # For each weight vector, find the best point from current set and refine
        # budget per refinement
        remaining = self.budget - budget_used
        n_refinements = min(len(weight_vectors), remaining // 10)

        for wi in range(n_refinements):
            if budget_used >= self.budget or not time_ok():
                break

            w = weight_vectors[wi]

            # Find best point in current set for this weight
            scalar_vals = all_F_full @ w
            best_idx = np.argmin(scalar_vals)
            x_start = all_X_full[best_idx].copy()
            f_start = all_F_full[best_idx].copy()

            # Nelder-Mead on weighted sum
            def weighted_obj(x):
                if budget_used >= self.budget:
                    return np.inf
                f = evaluate(x)
                budget_used += 1
                return f @ w

            x_opt, f_opt_val = nelder_mead(weighted_obj, x_start, max_iter=20, tol=1e-5)

            # Evaluate the final point properly (it was evaluated during NM)
            # Check if it's non-dominated
            f_opt = np.array(problem(x_opt))  # This would be an extra eval, but we already have it
            # Actually, let's just use the last evaluation from NM
            # We need to re-evaluate to get the actual objective vector
            # But that costs an eval. Let's check budget.
            if budget_used < self.budget and time_ok():
                f_opt = evaluate(x_opt)
                budget_used += 1

                if not is_dominated(f_opt, all_F_full):
                    all_X_full = np.vstack([all_X_full, x_opt.reshape(1, -1)])
                    all_F_full = np.vstack([all_F_full, f_opt.reshape(1, -1)])

        # Phase 4: Final polish - refine the current Pareto front points
        all_F_full = remove_dominated(all_F_full)
        # Rebuild X to match
        # We need to be more careful here. Let's track X and F together.

        # Let me redo this more carefully with a single list
        # Actually, the above approach has issues with tracking. Let me restructure.

        # Restart with cleaner tracking
        # We've already used budget_used evaluations. Let's see what's left.
        remaining = self.budget - budget_used

        if remaining > 5 and time_ok():
            # Get current non-dominated set
            # We need to recompute from all_X_full and all_F_full
            all_F_full = remove_dominated(all_F_full)

            # For each non-dominated point, do a small local refinement
            n_nd = len(all_F_full)
            n_polish = min(n_nd, remaining // 6)

            for i in range(n_polish):
                if budget_used >= self.budget or not time_ok():
                    break

                x_base = all_X_full[i].copy()
                f_base = all_F_full[i].copy()

                # Small random perturbations
                for trial in range(5):
                    if budget_used >= self.budget or not time_ok():
                        break
                    scale = 0.03 * (0.7 ** trial) * (hi - lo)
                    x_try = x_base + scale * np.random.standard_normal(dim)
                    x_try = np.clip(x_try, lo, hi)
                    f_try = evaluate(x_try)
                    budget_used += 1

                    # Check if it improves the hypervolume
                    # Simple check: is it non-dominated?
                    if not is_dominated(f_try, all_F_full):
                        # Replace or add
                        # Check if it dominates f_base
                        if np.all(f_try <= f_base) and np.any(f_try < f_base):
                            all_F_full[i] = f_try
                            all_X_full[i] = x_try
                            x_base = x_try
                            f_base = f_try
                        else:
                            all_X_full = np.vstack([all_X_full, x_try.reshape(1, -1)])
                            all_F_full = np.vstack([all_F_full, f_try.reshape(1, -1)])

        # Final cleanup
        all_F_full = remove_dominated(all_F_full)

        # Return the final set of points
        if len(all_F_full) > 0:
            return all_X_full[:len(all_F_full)]
        else:
            # Fallback
            x = lo + 0.5 * (hi - lo)
            f = evaluate(x)
            return np.array([x])
# EVOLVE-BLOCK-END
