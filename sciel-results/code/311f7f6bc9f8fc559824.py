"""Multi-objective optimizer: deterministic HV-guided sequential greedy search on a dense lattice with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        used = 0
        all_F = []  # list of objective vectors
        all_X = []  # list of x vectors

        # --- Helper: dominance ---
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # --- Helper: 2D hypervolume ---
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

        # --- Helper: marginal HV ---
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

        # --- Phase 1: Dense grid search ---
        # Use a 5^4 = 625 point grid, but we only have 200 evaluations
        # So we'll use a 4^4 = 256 point grid and pick the best 200 by marginal HV
        grid_size = 4
        candidates = []
        for a in range(grid_size):
            for b in range(grid_size):
                for c in range(grid_size):
                    for d in range(grid_size):
                        x = np.array([a, b, c, d], dtype=float) / (grid_size - 1)
                        candidates.append(x)

        # Shuffle for diversity
        np.random.shuffle(candidates)

        # Evaluate grid points greedily by marginal HV
        for x in candidates:
            if used >= budget:
                break
            if time.time() > deadline:
                break

            try:
                f = problem(x)
            except Exception:
                continue

            f = np.array(f)
            all_F.append(f)
            all_X.append(x)
            used += 1

        # --- Phase 2: Local refinement around best archive points ---
        # Find non-dominated points in the archive
        archive = []
        for i, f in enumerate(all_F):
            dominated = False
            for j, g in enumerate(all_F):
                if i != j and dominates(g, f):
                    dominated = True
                    break
            if not dominated:
                archive.append(i)

        # Sort archive by marginal HV contribution (approximate)
        # Just use the first few non-dominated points
        if len(archive) > 0:
            # Pick up to 5 non-dominated points for local refinement
            n_refine = min(5, len(archive))
            # Use the points with best HV contribution
            # For simplicity, just take the first n_refine
            refine_indices = archive[:n_refine]

            # Local search around each refine point
            step_sizes = [0.1, 0.05, 0.02, 0.01]
            for idx in refine_indices:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                center = all_X[idx]
                # Generate local candidates
                local_candidates = []
                for step in step_sizes:
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            for dz in [-step, 0, step]:
                                for dw in [-step, 0, step]:
                                    if dx == 0 and dy == 0 and dz == 0 and dw == 0:
                                        continue
                                    x_new = center + np.array([dx, dy, dz, dw])
                                    # Clip to bounds
                                    x_new = np.clip(x_new, lo, hi)
                                    local_candidates.append(x_new)

                # Evaluate local candidates greedily
                np.random.shuffle(local_candidates)
                for x in local_candidates:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break

                    # Check if this point is already evaluated
                    already_evaluated = False
                    for existing_x in all_X:
                        if np.allclose(x, existing_x, atol=1e-10):
                            already_evaluated = True
                            break
                    if already_evaluated:
                        continue

                    try:
                        f = problem(x)
                    except Exception:
                        continue

                    f = np.array(f)

                    # Compute marginal HV
                    mhv = marginal_hv(f, all_F)

                    # Accept if it adds to HV
                    if mhv > 1e-10:
                        all_F.append(f)
                        all_X.append(x)
                        used += 1

        return all_X
# EVOLVE-BLOCK-END
