"""Multi-objective optimizer: Sobol space-filling init + Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer: deterministic Sobol fill + local NM refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        # --- Helper functions ---

        def dominates(a, b):
            """Check if a dominates b."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Return indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            non_dom = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dom[i]:
                    continue
                for j in range(n):
                    if i == j or not non_dom[j]:
                        continue
                    if dominates(F[j], F[i]):
                        non_dom[i] = False
                        break
            return np.where(non_dom)[0]

        def is_valid(f):
            """Check if point is strictly better than reference in all objectives."""
            return np.all(f < ref)

        def hypervolume_3d(points, ref_pt):
            """Compute hypervolume for 3 objectives (exact for small sets)."""
            pts = np.array(points)
            pts = pts[np.all(pts < ref_pt, axis=1)]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts_s = pts[order]
            total_hv = 0.0
            for i in range(len(pts_s)):
                p = pts_s[i]
                # Project remaining points (from i onward) onto obj2-obj3 plane
                proj = pts_s[i:, 1:3]
                proj = proj[np.all(proj < ref_pt[1:], axis=1)]
                if len(proj) == 0:
                    continue
                # Sort by obj2
                proj_s = proj[np.argsort(proj[:, 0])]
                hv2 = 0.0
                max_o3 = 0.0
                for j in range(len(proj_s)):
                    q_pt = proj_s[j]
                    if q_pt[1] > max_o3:
                        max_o3 = q_pt[1]
                    hv2 += (ref_pt[1] - q_pt[0]) * (ref_pt[2] - max_o3)
                total_hv += (ref_pt[0] - p[0]) * hv2
            return total_hv

        # --- Phase 1: Sobol space-filling initialization ---
        n_init = min(60, max(20, budget // 3))
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        X_init_unit = sobol.random(n_init)
        X_init = lo + (hi - lo) * X_init_unit
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Build archive with only valid (non-dominated, better than ref) points
        archive_X = []
        archive_F = []
        for i in range(n_init):
            if is_valid(F_init[i]):
                archive_X.append(X_init[i])
                archive_F.append(F_init[i])

        # Remove dominated points from archive
        if archive_F:
            F_arr = np.array(archive_F)
            nd_idx = non_dominated_sort(F_arr)
            archive_X = [archive_X[i] for i in nd_idx]
            archive_F = [F_arr[i] for i in nd_idx]

        # --- Phase 2: Local refinement with Nelder-Mead on top-k non-dominated points ---
        # Select top-k points from the non-dominated archive for local search
        # We want to explore the Pareto front, so pick diverse points
        if archive_F:
            F_arr = np.array(archive_F)
            # Score each point by its contribution to hypervolume (approximate: distance to worst)
            # Simpler: just pick all non-dominated points, up to k
            k = min(len(archive_F), 15)
            # Pick evenly spaced from the sorted non-dominated set
            order = np.argsort(F_arr[:, 0])
            selected_indices = np.linspace(0, len(order) - 1, k, dtype=int)
            selected_indices = order[selected_indices]
        else:
            selected_indices = []

        # For each selected point, run Nelder-Mead on a scalarized objective
        # Use weighted sum with different weight vectors to explore different parts of the front
        # Weight vectors for 3 objectives
        weight_vectors = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2],
            [0.2, 0.4, 0.4],
            [0.4, 0.2, 0.4],
            [0.33, 0.33, 0.34],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ])

        n_weights = len(weight_vectors)
        nm_budget_per = max(1, (budget - used) // (len(selected_indices) * n_weights + 1))
        nm_budget_per = min(nm_budget_per, 30)  # cap per NM run

        # Create bounds for scipy minimize
        bounds_scipy = list(zip(lo, hi))

        for idx in selected_indices:
            if used >= budget:
                break
            if time.time() > deadline:
                break

            x0 = archive_X[idx] if idx < len(archive_X) else (lo + hi) / 2

            for w in weight_vectors:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Scalarized objective: minimize weighted sum
                def scalar_obj(x, w=w):
                    f = problem(x)
                    return np.dot(f, w)

                # Use Nelder-Mead with limited iterations
                try:
                    result = minimize(
                        scalar_obj,
                        x0,
                        method='Nelder-Mead',
                        bounds=bounds_scipy,
                        options={
                            'maxiter': nm_budget_per,
                            'xatol': 1e-4,
                            'fatol': 1e-4,
                            'adaptive': True
                        }
                    )
                    # Count evaluations used (approximate: result.nfev)
                    nfev = result.nfev if hasattr(result, 'nfev') else nm_budget_per
                    used += nfev

                    x_opt = result.x
                    f_opt = problem(x_opt)
                    used += 1  # one extra evaluation for recording

                    if is_valid(f_opt):
                        # Add to archive and remove dominated
                        new_F_arr = np.array(archive_F + [f_opt])
                        new_X_arr = archive_X + [x_opt]
                        nd_idx = non_dominated_sort(new_F_arr)
                        archive_X = [new_X_arr[i] for i in nd_idx]
                        archive_F = [new_F_arr[i] for i in nd_idx]
                except Exception:
                    pass

        # --- Phase 3: Use remaining budget for additional Sobol sampling in promising regions ---
        if used < budget:
            # Sample around the current Pareto front
            n_remaining = budget - used
            n_extra = min(n_remaining, 40)

            # Generate points near the archive
            if archive_F:
                F_arr = np.array(archive_F)
                X_arr = np.array(archive_X)
                # Center around mean of archive
                center = np.mean(X_arr, axis=0)
                spread = np.std(X_arr, axis=0) + 0.1

                for _ in range(n_extra):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    x_new = center + spread * (np.random.randn(dim) * 0.5)
                    x_new = np.clip(x_new, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    if is_valid(f_new):
                        new_F_arr = np.array(archive_F + [f_new])
                        new_X_arr = archive_X + [x_new]
                        nd_idx = non_dominated_sort(new_F_arr)
                        archive_X = [new_X_arr[i] for i in nd_idx]
                        archive_F = [new_F_arr[i] for i in nd_idx]

        # Return the archive (list of x points)
        return archive_X
# EVOLVE-BLOCK-END
