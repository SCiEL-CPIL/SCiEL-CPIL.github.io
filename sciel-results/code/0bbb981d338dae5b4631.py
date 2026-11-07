"""Tchebycheff-based multi-objective optimizer with dense weight grid and multi-start NM."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using Tchebycheff scalarization with dense weight grid."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        used = 0
        archive_X = []
        archive_F = []

        # --- Helper: dominance ---
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # --- Helper: 2D hypervolume (for slices in 3D) ---
        def hv_2d(pts, ref_y, ref_z):
            if len(pts) == 0:
                return 0.0
            pts = np.array(pts)
            mask = (pts[:, 0] < ref_y) & (pts[:, 1] < ref_z)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n_pts = len(pts)
            hv2 = 0.0
            for i in range(n_pts):
                y_next = pts[i + 1, 0] if i + 1 < n_pts else ref_y
                if i + 1 < n_pts:
                    min_z_remaining = np.min(pts[i + 1:, 1])
                else:
                    min_z_remaining = ref_z
                if min_z_remaining > pts[i, 1]:
                    w_y = y_next - pts[i, 0]
                    w_z = min_z_remaining - pts[i, 1]
                    if w_y > 0 and w_z > 0:
                        hv2 += w_y * w_z
            return hv2

        # --- Helper: 3D hypervolume ---
        def hv_3d(points, ref):
            if len(points) == 0:
                return 0.0
            P = np.array(points)
            mask = np.all(P < ref, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0

            order = np.argsort(P[:, 0])
            P_sorted = P[order]
            n_pts = len(P_sorted)
            hv = 0.0
            for i in range(n_pts):
                f0_min = P_sorted[i, 0]
                f0_max = P_sorted[i + 1, 0] if i + 1 < n_pts else ref[0]
                if f0_max <= f0_min:
                    continue
                mask_slice = (P_sorted[:, 0] >= f0_min) & (P_sorted[:, 0] < f0_max)
                slice_pts = P_sorted[mask_slice, 1:3]
                hv_slice = hv_2d(slice_pts, ref[1], ref[2])
                hv += hv_slice * (f0_max - f0_min)
            return hv

        # --- Helper: marginal HV contribution of a new point ---
        def marginal_hv(point, archive_F):
            if not np.all(point < ref_point):
                return 0.0
            if len(archive_F) == 0:
                return hv_3d([point], ref_point)
            for f in archive_F:
                if dominates(f, point):
                    return 0.0
            hv_before = hv_3d(archive_F, ref_point)
            archive_new = list(archive_F) + [point]
            hv_after = hv_3d(archive_new, ref_point)
            return hv_after - hv_before

        # --- Helper: update archive with new point ---
        def update_archive(x, f):
            nonlocal archive_X, archive_F
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            dominated = False
            for af in new_archive_F:
                if dominates(af, f):
                    dominated = True
                    break
            if not dominated:
                new_archive_X.append(x)
                new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

        # --- Generate dense Tchebycheff weight vectors ---
        # Use a grid on the 2-simplex for 3 objectives
        n_levels = 8  # number of divisions along each axis of the simplex
        weight_vectors = []
        for i in range(n_levels + 1):
            for j in range(n_levels + 1 - i):
                k = n_levels - i - j
                w = np.array([i, j, k], dtype=float)
                w = w / w.sum()
                weight_vectors.append(w)

        # Add some random weights for extra coverage
        n_random_weights = 20
        for _ in range(n_random_weights):
            w = rng.random(n_obj)
            w = w / w.sum()
            weight_vectors.append(w)

        # --- Phase 1: Multi-start Nelder-Mead on Tchebycheff scalarization ---
        # For each weight vector, run Nelder-Mead from multiple starting points
        n_starts_per_weight = 3

        # Precompute some starting points
        start_points = []
        # Vertices of the hypercube
        for i in range(2 ** dim):
            pt = np.array([(i >> d) & 1 for d in range(dim)], dtype=float)
            start_points.append(lo + pt * span)
        # Center
        start_points.append(lo + span * 0.5)
        # Random points
        for _ in range(10):
            start_points.append(lo + rng.random(dim) * span)

        for w in weight_vectors:
            if used >= budget:
                break
            if time.time() > deadline - 20:
                break

            # Tchebycheff scalarization: min max_i w_i * |f_i(x) - z_i^*|
            # We don't know z^*, so use a simple weighted sum as proxy
            # Actually, use weighted sum: min sum w_i * f_i(x)
            for start in start_points[:n_starts_per_weight]:
                if used >= budget:
                    break
                if time.time() > deadline - 10:
                    break

                def scalarized(x):
                    try:
                        f_val = problem(x)
                        f_val = np.array(f_val, dtype=float)
                        return np.dot(w, f_val)
                    except Exception:
                        return 1e10

                try:
                    result = minimize(
                        scalarized,
                        start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 80,
                            'xatol': 1e-5,
                            'fatol': 1e-7,
                            'adaptive': True
                        }
                    )
                except Exception:
                    continue

                x_opt = np.clip(result.x, lo, hi)
                try:
                    f_opt = problem(x_opt)
                    f_opt = np.array(f_opt, dtype=float)
                except Exception:
                    continue

                mhv = marginal_hv(f_opt, archive_F)
                if mhv > 1e-10:
                    update_archive(x_opt, f_opt)
                    used += 1

        # --- Phase 2: Refine around best archive points with more weight vectors ---
        if len(archive_F) > 0 and used < budget:
            # Select top archive points by HV contribution
            hv_contribs = []
            for i, f in enumerate(archive_F):
                archive_without = [archive_F[j] for j in range(len(archive_F)) if j != i]
                hv_without = hv_3d(archive_without, ref_point)
                hv_with = hv_3d(archive_F, ref_point)
                hv_contribs.append(hv_with - hv_without)

            order = np.argsort(hv_contribs)[::-1]
            n_refine = min(10, len(archive_F))

            for idx in order[:n_refine]:
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                center_x = archive_X[idx]

                # Try more refined weight vectors around this point
                local_weights = [
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                    np.array([0.5, 0.5, 0.0]),
                    np.array([0.5, 0.0, 0.5]),
                    np.array([0.0, 0.5, 0.5]),
                    np.array([1.0/3, 1.0/3, 1.0/3]),
                    np.array([0.7, 0.15, 0.15]),
                    np.array([0.15, 0.7, 0.15]),
                    np.array([0.15, 0.15, 0.7]),
                ]

                for lw in local_weights:
                    if used >= budget:
                        break
                    if time.time() > deadline - 2:
                        break

                    def scalarized(x):
                        try:
                            f_val = problem(x)
                            f_val = np.array(f_val, dtype=float)
                            return np.dot(lw, f_val)
                        except Exception:
                            return 1e10

                    try:
                        result = minimize(
                            scalarized,
                            center_x,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 100,
                                'xatol': 1e-6,
                                'fatol': 1e-8,
                                'adaptive': True
                            }
                        )
                    except Exception:
                        continue

                    x_opt = np.clip(result.x, lo, hi)
                    try:
                        f_opt = problem(x_opt)
                        f_opt = np.array(f_opt, dtype=float)
                    except Exception:
                        continue

                    mhv = marginal_hv(f_opt, archive_F)
                    if mhv > 1e-10:
                        update_archive(x_opt, f_opt)
                        used += 1

        # --- Phase 3: Final micro-refinement ---
        if used < budget and time.time() < deadline - 2:
            polish_step = 0.005
            n_polish = min(30, budget - used)
            for _ in range(n_polish):
                if time.time() > deadline - 1:
                    break
                if used >= budget:
                    break

                if len(archive_X) > 0:
                    # Select archive point with best marginal HV
                    hv_contribs = []
                    for i, f in enumerate(archive_F):
                        archive_without = [archive_F[j] for j in range(len(archive_F)) if j != i]
                        hv_without = hv_3d(archive_without, ref_point)
                        hv_with = hv_3d(archive_F, ref_point)
                        hv_contribs.append(hv_with - hv_without)
                    best_idx = int(np.argmax(hv_contribs))
                    parent_x = archive_X[best_idx]
                else:
                    parent_x = lo + span * 0.5

                noise = polish_step * span * rng.standard_normal(dim)
                child_x = np.clip(parent_x + noise, lo, hi)
                try:
                    child_f = problem(child_x)
                    child_f = np.array(child_f, dtype=float)
                except Exception:
                    continue

                mhv = marginal_hv(child_f, archive_F)
                if mhv > 1e-10:
                    update_archive(child_x, child_f)
                    used += 1

        # Return the archive points
        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([lo + span * 0.5])
# EVOLVE-BLOCK-END
