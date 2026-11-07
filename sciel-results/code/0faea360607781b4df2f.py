"""Multi-objective optimizer for RE34 with hypervolume-aware evolutionary search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 (5 variables, 3 objectives, bounds [1,3])."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_3d(self, pts, ref):
        """Compute 3D hypervolume using inclusion-exclusion via 2D slices."""
        if len(pts) == 0:
            return 0.0
        m = np.all(pts < ref, axis=1)
        p = pts[m]
        if len(p) == 0:
            return 0.0
        # Sort by first objective
        p = p[np.argsort(p[:, 0])]
        n = len(p)
        hv = 0.0
        for i in range(n):
            next_f0 = p[i + 1, 0] if i + 1 < n else ref[0]
            width = next_f0 - p[i, 0]
            # 2D hypervolume of remaining points projected onto dims 1,2
            sub = p[i:, 1:]
            ref2 = ref[1:]
            # Filter points better than ref in both dims
            mask2 = np.all(sub < ref2, axis=1)
            sub = sub[mask2]
            if len(sub) == 0:
                continue
            sub = sub[np.argsort(sub[:, 0])]
            area = 0.0
            for j in range(len(sub)):
                next_x = sub[j + 1, 0] if j + 1 < len(sub) else ref2[0]
                w2 = next_x - sub[j, 0]
                h2 = ref2[1] - sub[j, 1]
                if w2 > 0 and h2 > 0:
                    area += w2 * h2
            hv += width * area
        return hv

    def _hv_contributions(self, pts, ref):
        """Compute per-point hypervolume contributions for 3D points (fast version)."""
        n = len(pts)
        if n == 0:
            return np.zeros(n)
        mask = np.all(pts < ref, axis=1)
        if not np.any(mask):
            return np.zeros(n)
        pts_f = pts[mask]
        idx = np.where(mask)[0]
        m = len(pts_f)
        contrib = np.zeros(n)
        
        # Sort by first objective for efficient contribution calculation
        order = np.argsort(pts_f[:, 0])
        pts_sorted = pts_f[order]
        idx_sorted = idx[order]
        
        full_hv = self._hv_3d(pts_f, ref)
        
        # For each point, compute contribution by finding the volume it uniquely covers
        # A point's contribution is the volume of the box from the point to the reference
        # that is not covered by any other point that dominates it in the relevant dimensions
        for k in range(m):
            i = order[k]
            # Find all points that are better in at least one objective and no worse in others
            # that could "block" this point's contribution
            # Simple approach: contribution = full_hv - hv_without_point
            # But that's O(m^2). Let's use a faster heuristic.
            
            # Heuristic: the contribution is approximately the volume of the box
            # from this point to ref, minus boxes from points that dominate it
            box_vol = 1.0
            for d in range(self.n_obj):
                box_vol *= (ref[d] - pts_f[i, d])
            
            # Subtract contributions from points that dominate this one
            # A point j dominates i if F[j] <= F[i] in all dims and < in at least one
            for j in range(m):
                if j == i:
                    continue
                if np.all(pts_f[j] <= pts_f[i]) and np.any(pts_f[j] < pts_f[i]):
                    # j dominates i, so i's contribution is reduced
                    # Approximate reduction
                    box_vol *= 0.5
                    break  # Conservative: just halve it
            
            contrib[idx[k]] = max(box_vol, 0.0)
        
        return contrib

    def _non_dominated_mask(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return ~dominated

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        all_X = []
        all_F = []
        used = 0

        # Phase 1: Diverse random initialization with larger population
        pop_size = 40
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used += pop_size
        all_X.extend(X.tolist())
        all_F.extend(F.tolist())

        # Phase 2: Differential Evolution with binomial crossover
        CR = 0.7  # Crossover rate
        F_factor = 0.5  # Differential weight
        pop = len(F)

        while used < self.budget:
            if time.time() > deadline:
                break

            # Compute non-dominated front
            nd_mask = self._non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]

            # Select victim (worst dominated or lowest HV contribution among non-dominated)
            dom_mask = ~nd_mask
            dom_indices = np.where(dom_mask)[0]
            if len(dom_indices) > 0:
                victim_idx = dom_indices[int(np.argmax(np.sum(F[dom_indices], axis=1)))]
            else:
                # All non-dominated: use crowding distance or HV contribution
                contribs = self._hv_contributions(F, ref)
                nd_contribs = contribs[nd_indices]
                min_pos = np.argmin(nd_contribs)
                victim_idx = nd_indices[min_pos]

            # Select three distinct parents for DE (different from victim)
            candidates = [i for i in range(pop) if i != victim_idx]
            if len(candidates) < 3:
                # Fallback to simple mutation if population too small
                parent_idx = rng.randint(pop)
                parent = X[parent_idx].copy()
                progress = used / self.budget
                step = (0.4 - 0.37 * progress) * (hi - lo)
                child = parent + step * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
            else:
                # Select 3 distinct parents
                parent_indices = rng.choice(candidates, size=3, replace=False)
                r1, r2, r3 = parent_indices
                
                # Differential evolution mutation
                v = X[r1] + F_factor * (X[r2] - X[r3])
                
                # Binomial crossover
                jrand = rng.randint(self.dim)
                child = X[victim_idx].copy()
                for j in range(self.dim):
                    if rng.random() < CR or j == jrand:
                        child[j] = v[j]
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
