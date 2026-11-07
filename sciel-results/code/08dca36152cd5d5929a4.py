"""Multi-objective optimizer on RE34 with hypervolume-driven search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HV-guided multi-objective optimizer for RE34 (5 vars, 3 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        ref = np.array([1705.0, 12.0, 0.3])

        def is_better_than_ref(f):
            return np.all(f < ref)

        def hv_2d(points_2d, ref_2d):
            """Fast 2D hypervolume for points < ref_2d. O(n log n)."""
            m = len(points_2d)
            if m == 0:
                return 0.0
            if m == 1:
                return float((ref_2d[0] - points_2d[0, 0]) * (ref_2d[1] - points_2d[0, 1]))
            pts2 = points_2d[np.argsort(points_2d[:, 0])]
            hv = 0.0
            min_y = ref_2d[1]
            for i in range(m - 1, -1, -1):
                p = pts2[i]
                if p[1] < min_y:
                    min_y = p[1]
                x_next = pts2[i + 1, 0] if i + 1 < m else ref_2d[0]
                hv += (x_next - p[0]) * (ref_2d[1] - min_y)
            return hv

        def hv_3d(points, ref_point):
            """Fast 3D hypervolume using slice method. O(n^2) but with small constant."""
            n = len(points)
            if n == 0:
                return 0.0
            if n == 1:
                return float(np.prod(ref_point - points[0]))
            pts = points[np.argsort(points[:, 0])]
            hv = 0.0
            for i in range(n):
                x_i = pts[i, 0]
                x_next = pts[i + 1, 0] if i + 1 < n else ref_point[0]
                yz = pts[:i + 1, 1:3]
                hv += (x_next - x_i) * hv_2d(yz, ref_point[1:3])
            return hv

        def is_pareto(F_arr):
            """Return mask of Pareto-optimal points."""
            n = len(F_arr)
            if n == 0:
                return np.array([], dtype=bool)
            pareto = np.ones(n, dtype=bool)
            for i in range(n):
                if not pareto[i]:
                    continue
                for j in range(n):
                    if i == j or not pareto[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        pareto[i] = False
                        break
            return pareto

        def hv_contrib_single(point, pareto_set, ref_point):
            """Compute HV contribution of a single point given the Pareto set."""
            if len(pareto_set) == 0:
                return float(np.prod(ref_point - point))
            with_pt = np.vstack([pareto_set, [point]])
            mask_with = np.all(with_pt < ref_point, axis=1)
            with_pt = with_pt[mask_with]
            mask_without = np.all(pareto_set < ref_point, axis=1)
            without_pt = pareto_set[mask_without]
            hv_with = hv_3d(with_pt, ref_point) if len(with_pt) > 0 else 0.0
            hv_without = hv_3d(without_pt, ref_point) if len(without_pt) > 0 else 0.0
            return hv_with - hv_without

        rng = np.random.RandomState(42)

        # Initial population: Latin Hypercube Sampling
        pop_size = min(40, budget)
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + 0.5) / pop_size
            X[:, d] = np.clip(X[:, d], lo[d], hi[d])

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Filter valid points
        valid_mask = np.array([is_better_than_ref(f) for f in F])
        pareto_X = X[valid_mask].copy()
        pareto_F = F[valid_mask].copy()

        # Cache HV contributions
        hv_cache = {}
        pareto_version = 0

        def compute_all_hv_contribs(pareto_F_arr):
            """Compute HV contribution for each point in Pareto set."""
            n = len(pareto_F_arr)
            if n == 0:
                return np.array([])
            contribs = np.zeros(n)
            for i in range(n):
                others = np.delete(pareto_F_arr, i, axis=0)
                contribs[i] = hv_contrib_single(pareto_F_arr[i], others, ref)
            return contribs

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Get current Pareto set
            if len(pareto_F) > 0:
                p_mask = is_pareto(pareto_F)
                pareto_F_cur = pareto_F[p_mask]
                pareto_X_cur = pareto_X[p_mask]
            else:
                pareto_F_cur = pareto_F
                pareto_X_cur = pareto_X

            # Select parent based on HV contribution
            if len(pareto_F_cur) > 0:
                contribs = compute_all_hv_contribs(pareto_F_cur)
                contribs = np.clip(contribs, 1e-10, None)
                # Use temperature-based selection for exploration
                progress = used / budget
                temperature = 0.5 * (1 - progress) + 0.1
                probs = np.exp(contribs / temperature)
                probs = probs / probs.sum()
                parent_idx = rng.choice(len(pareto_F_cur), p=probs)
                parent = pareto_X_cur[parent_idx]
            else:
                parent = X[rng.randint(len(X))]

            # Adaptive mutation with multiple scales
            progress = used / budget
            base_step = 0.3 * (1 - progress) + 0.05
            # Occasionally use larger steps for exploration
            if rng.random() < 0.1:
                step_scale = base_step * 3.0
            else:
                step_scale = base_step

            # Gaussian mutation
            child = parent + step_scale * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update archives
            if is_better_than_ref(fc):
                pareto_X = np.vstack([pareto_X, [child]])
                pareto_F = np.vstack([pareto_F, [fc]])
                X = np.vstack([X, [child]])
                F = np.vstack([F, [fc]])

            # Prune Pareto set periodically to keep it manageable
            if len(pareto_F) > 80:
                p_mask = is_pareto(pareto_F)
                pareto_X = pareto_X[p_mask]
                pareto_F = pareto_F[p_mask]
                if len(pareto_F) > 60:
                    contribs = compute_all_hv_contribs(pareto_F)
                    top_idx = np.argsort(contribs)[-60:]
                    pareto_X = pareto_X[top_idx]
                    pareto_F = pareto_F[top_idx]

        return X

# EVOLVE-BLOCK-END
