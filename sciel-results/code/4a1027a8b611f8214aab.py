"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer combining random sampling, L-BFGS-B local search,
    and hypervolume-aware evolutionary selection for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, pts, ref2):
        """Compute 2D hypervolume. pts: (N,2), ref2: (2,)."""
        if len(pts) == 0:
            return 0.0
        m = np.all(pts < ref2, axis=1)
        p = pts[m]
        if len(p) == 0:
            return 0.0
        p = p[np.argsort(p[:, 0])]
        n = len(p)
        vol = 0.0
        for i in range(n):
            next_x = p[i + 1, 0] if i + 1 < n else ref2[0]
            width = next_x - p[i, 0]
            height = ref2[1] - p[i, 1]
            vol += width * height
        return vol

    def _hv_3d(self, pts, ref):
        """Compute 3D hypervolume. pts: (N,3), ref: (3,)."""
        if len(pts) == 0:
            return 0.0
        m = np.all(pts < ref, axis=1)
        p = pts[m]
        if len(p) == 0:
            return 0.0
        p = p[np.argsort(p[:, 0])]
        n = len(p)
        hv = 0.0
        for i in range(n):
            next_f0 = p[i + 1, 0] if i + 1 < n else ref[0]
            width = next_f0 - p[i, 0]
            area = self._hv_2d(p[i:, 1:], ref[1:])
            hv += width * area
        return hv

    def _hv_contributions(self, pts, ref):
        """Compute per-point hypervolume contributions for 3D points."""
        n = len(pts)
        if n == 0:
            return np.array([])
        mask = np.all(pts < ref, axis=1)
        if not np.any(mask):
            return np.zeros(n)
        pts_f = pts[mask]
        idx = np.where(mask)[0]
        m = len(pts_f)
        contrib = np.zeros(n)
        full_hv = self._hv_3d(pts_f, ref)
        for k in range(m):
            mask_remove = np.ones(m, dtype=bool)
            mask_remove[k] = False
            hv_without = self._hv_3d(pts_f[mask_remove], ref)
            contrib[idx[k]] = full_hv - hv_without
        return contrib

    def _non_dominated_mask(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return ~dominated

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit

        rng = np.random  # Use global RNG (seeded by evaluator)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            all_X.append(x)
            all_F.append(f)
            return x, f

        # Phase 1: Random initialization
        pop_size = 20
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used += pop_size
        all_X.extend(X.tolist())
        all_F.extend(F.tolist())

        # Phase 2: L-BFGS-B local search on individual objectives from random starts
        # Use a portion of the budget for gradient-based refinement
        local_search_budget = 40
        local_starts = 10
        obj_weights = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.3, 0.2],
            [0.2, 0.3, 0.5],
            [0.3, 0.5, 0.2],
            [0.4, 0.4, 0.2],
            [0.2, 0.4, 0.4],
            [0.4, 0.2, 0.4],
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
        ]

        for i in range(local_starts):
            if used >= self.budget or time.time() > deadline:
                break
            if used + local_search_budget > self.budget:
                break

            w = obj_weights[i]
            x0 = lo + (hi - lo) * rng.random(self.dim)

            def objective(x):
                f = problem(x)
                return np.dot(w, f)

            try:
                res = minimize(
                    objective, x0, method='L-BFGS-B',
                    bounds=list(bounds),
                    options={'maxiter': 50, 'maxfun': local_search_budget}
                )
                if res.success or res.fun < np.inf:
                    eval_point(res.x)
            except Exception:
                pass

        # Phase 3: Evolutionary search with hypervolume-aware selection
        # Build initial population from all evaluated points
        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        # Select best 20 points for population
        if len(all_F_arr) > 20:
            # Keep non-dominated points first, then fill with diverse points
            nd_mask = self._non_dominated_mask(all_F_arr)
            nd_count = np.sum(nd_mask)
            if nd_count >= 20:
                # Select 20 from non-dominated
                nd_indices = np.where(nd_mask)[0]
                # Use crowding-like diversity: pick spread out points
                # Simple approach: sort by sum of objectives, pick evenly
                sums = np.sum(all_F_arr[nd_mask], axis=1)
                order = np.argsort(sums)
                # Pick first, last, and evenly spaced
                step = max(1, len(order) // 20)
                selected = [order[i * step] for i in range(20) if i * step < len(order)]
                while len(selected) < 20:
                    selected.append(order[len(selected) % len(order)])
                sel_indices = nd_indices[selected[:20]]
            else:
                sel_indices = np.where(nd_mask)[0]
                remaining = np.where(~nd_mask)[0]
                while len(sel_indices) < 20 and len(remaining) > 0:
                    # Pick most diverse remaining point
                    if len(sel_indices) == 0:
                        pick = remaining[0]
                    else:
                        # Pick point farthest from existing selected
                        dists = np.min(
                            np.sum((all_F_arr[remaining][:, None, :] - all_F_arr[sel_indices][None, :, :]) ** 2, axis=2),
                            axis=1
                        )
                        pick = remaining[np.argmax(dists)]
                    sel_indices = np.append(sel_indices, pick)
                    remaining = np.delete(remaining, np.where(remaining == pick)[0])
                sel_indices = sel_indices[:20]
        else:
            sel_indices = np.arange(len(all_F_arr))

        X = all_X_arr[sel_indices].copy()
        F = all_F_arr[sel_indices].copy()

        # Evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Compute non-dominated front
            nd_mask = self._non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                parent_idx = rng.integers(len(F))
                victim_idx = int(np.argmax(np.sum(F, axis=1)))
            else:
                # Select parent from non-dominated front
                parent_idx = nd_indices[rng.integers(len(nd_indices))]

                # Select victim: dominated point with worst sum, or non-dominated with lowest HV contribution
                dom_mask = ~nd_mask
                dom_indices = np.where(dom_mask)[0]
                if len(dom_indices) > 0:
                    victim_idx = dom_indices[int(np.argmax(np.sum(F[dom_indices], axis=1)))]
                else:
                    # All non-dominated: remove point with lowest HV contribution
                    contribs = self._hv_contributions(F, ref)
                    # Only consider non-dominated points
                    nd_contribs = contribs[nd_indices]
                    min_pos = np.argmin(nd_contribs)
                    victim_idx = nd_indices[min_pos]

            parent = X[parent_idx].copy()

            # Adaptive mutation
            progress = used / self.budget
            step = 0.12 * (1.0 - 0.4 * progress) * (hi - lo)

            # Gaussian mutation
            child = parent + step * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            all_X.append(child)
            all_F.append(fc)

            # Replace victim in population
            X[victim_idx] = child
            F[victim_idx] = fc

        return all_X

# EVOLVE-BLOCK-END
