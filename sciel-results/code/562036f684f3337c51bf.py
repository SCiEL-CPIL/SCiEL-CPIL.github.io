# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated(self, F):
        """Return boolean mask of non-dominated points (minimization)."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        if n == 0:
            return nd
        for i in range(n):
            if not nd[i]:
                continue
            # Check if i is dominated by any j
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return nd

    def _hv_2d(self, pts, ref2):
        """Exact 2D hypervolume for a set of points (minimization)."""
        if len(pts) == 0:
            return 0.0
        p = pts[np.all(pts < ref2, axis=1)]
        if len(p) == 0:
            return 0.0
        # Sort by first coordinate (y)
        p = p[np.argsort(p[:, 0])]
        n = len(p)
        hv = 0.0
        min_z = np.inf
        for i in range(n):
            min_z = min(min_z, p[i, 1])
            y_next = p[i + 1, 0] if i < n - 1 else ref2[0]
            w = y_next - p[i, 0]
            h = ref2[1] - min_z
            if w > 0 and h > 0:
                hv += w * h
        return hv

    def _hv_3d(self, pts, ref):
        """Exact 3D hypervolume for a set of points (minimization)."""
        if len(pts) == 0:
            return 0.0
        p = pts[np.all(pts < ref, axis=1)]
        if len(p) == 0:
            return 0.0
        p = p[np.argsort(p[:, 0])]
        n = len(p)
        hv = 0.0
        for i in range(n):
            x_next = p[i + 1, 0] if i < n - 1 else ref[0]
            w = x_next - p[i, 0]
            if w <= 0:
                continue
            hv_2d_val = self._hv_2d(p[:i + 1, 1:3], ref[1:3])
            hv += w * hv_2d_val
        return hv

    def _hv_contributions(self, pts, ref):
        """Compute HV contribution of each point via remove-and-recompute."""
        n = len(pts)
        contrib = np.zeros(n)
        mask = np.all(pts < ref, axis=1)
        if not np.any(mask):
            return contrib
        total = self._hv_3d(pts, ref)
        for i in range(n):
            if not mask[i]:
                continue
            keep = mask.copy()
            keep[i] = False
            hv_without = self._hv_3d(pts[keep], ref)
            contrib[i] = total - hv_without
        return contrib

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin below 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([1705.0, 12.0, 0.3])
        dim = self.dim

        # Use global numpy random which respects np.random.seed set by evaluator
        # Initialize population with random points
        pop_size = 30
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Keep all evaluated points for final return
        all_X = list(X)
        all_F = list(F)

        while used < self.budget:
            if time.time() > deadline:
                break

            # Compute non-dominated set
            nd_mask = self._non_dominated(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) == 0:
                parent_idx = np.random.randint(pop_size)
                victim_idx = int(np.argmax(np.sum(F, axis=1)))
            else:
                # Select parent: weighted by inverse normalized sum
                F_nd = F[nd_idx]
                # Use simple random from non-dominated, or bias toward "central"
                # Random is fine for diversity
                parent_idx = nd_idx[np.random.randint(len(nd_idx))]

                # Select victim
                dom_idx = np.where(~nd_mask)[0]
                if len(dom_idx) > 0:
                    # Remove worst dominated
                    victim_idx = dom_idx[int(np.argmax(np.sum(F[dom_idx], axis=1)))]
                else:
                    # All non-dominated: remove lowest HV contribution
                    contribs = self._hv_contributions(F, ref)
                    victim_idx = int(np.argmin(contribs))

            parent = X[parent_idx]

            # Adaptive mutation scale: start wide, narrow down
            progress = used / self.budget
            # Smooth decay from 0.3 to 0.04
            scale = 0.3 - 0.26 * progress
            # Also use a local refinement around good points occasionally
            if np.random.random() < 0.15:
                # Local fine search around parent
                scale = 0.05 * (1.0 - progress * 0.5)

            child = parent + scale * (hi - lo) * np.random.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            all_X.append(child)
            all_F.append(fc)

            X[victim_idx] = child
            F[victim_idx] = fc

        # Return all evaluated points (the evaluator records every returned objective vector)
        return np.array(all_X)

# EVOLVE-BLOCK-END
