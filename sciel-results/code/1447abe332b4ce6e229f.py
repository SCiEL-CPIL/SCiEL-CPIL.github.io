"""Multi-objective optimizer for RE34 combining MOEA/D decomposition with hypervolume-aware selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE34 (5 variables, 3 objectives, bounds [1,3]).
    
    Combines:
    - MOEA/D-style weight decomposition for systematic Pareto front coverage
    - Hypervolume-aware evolutionary selection for diversity maintenance
    - L-BFGS-B local search for gradient-based refinement
    """

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

    def _generate_weights(self, n_sub):
        """Generate n_sub weight vectors on the simplex for 3 objectives."""
        weights = []
        # Use a grid-based approach on the 2D simplex
        # For 3 objectives, weights (w1, w2, w3) with w1+w2+w3=1, wi>=0
        # We can parameterize by (w1, w2) with w3 = 1 - w1 - w2
        n_per_dim = int(np.ceil(n_sub ** 0.5))
        for i in range(n_per_dim):
            w1 = i / (n_per_dim - 1) if n_per_dim > 1 else 0.0
            for j in range(n_per_dim - i):
                w2 = j / (n_per_dim - 1) if n_per_dim > 1 else 0.0
                w3 = 1.0 - w1 - w2
                if w3 >= -1e-10:
                    w = np.array([w1, w2, max(0, w3)])
                    # Normalize to ensure sum is exactly 1
                    s = w.sum()
                    if s > 1e-10:
                        w = w / s
                    weights.append(w)
        
        # If we have too many, sample randomly
        while len(weights) < n_sub:
            w = np.random.dirichlet(np.ones(3))
            weights.append(w)
        
        # If we have too few, add random ones
        if len(weights) > n_sub:
            # Keep evenly spaced ones
            weights = weights[:n_sub]
        
        return np.array(weights)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        # Use global RNG (seeded by evaluator) for reproducibility
        rng = np.random
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f

        # Phase 1: Diverse random initialization (20 points)
        pop_size = 20
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([eval_point(x) for x in X])

        # Phase 2: MOEA/D-style weight-based local search
        # Generate weight vectors for 3 objectives
        n_sub = 30
        weights = self._generate_weights(n_sub)
        
        # For each weight, run a short L-BFGS-B on the weighted sum
        # from a diverse starting point
        for w in weights:
            if time.time() > deadline or used >= self.budget:
                break
            
            # Pick a random starting point
            x0 = lo + (hi - lo) * rng.random(self.dim)
            
            def weighted_sum(x):
                f = np.array(problem(x))
                return np.dot(w, f)
            
            try:
                res = minimize(
                    weighted_sum, x0, method='L-BFGS-B',
                    bounds=list(bounds),
                    options={'maxiter': 10, 'gtol': 1e-3}
                )
                if res.success or res.fun < 1e10:
                    x_opt = np.clip(res.x, lo, hi)
                    f_opt = np.array(problem(x_opt))
                    if not any(np.allclose(x_opt, prev) for prev in all_X):
                        eval_point(x_opt)
            except Exception:
                pass

        # Phase 3: Evolutionary refinement with hypervolume-aware selection
        # Rebuild population from best points found so far
        if len(all_F) > 0:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            
            # Filter valid points (better than reference)
            valid_mask = np.all(F_arr < ref, axis=1)
            if np.any(valid_mask):
                valid_idx = np.where(valid_mask)[0]
                # Sort by sum of objectives
                sums = F_arr[valid_mask].sum(axis=1)
                sorted_idx = valid_idx[np.argsort(sums)]
                
                # Take top 20 for population
                n_pop = min(pop_size, len(sorted_idx))
                pop_idx = sorted_idx[:n_pop]
                X = X_arr[pop_idx].copy()
                F = F_arr[pop_idx].copy()
            else:
                # If no valid points, use all
                X = X_arr.copy()
                F = F_arr.copy()
        
        # Continue evolutionary search
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Compute non-dominated front
            nd_mask = self._non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                parent_idx = rng.randint(len(F))
                victim_idx = int(np.argmax(np.sum(F, axis=1)))
            else:
                # Select parent from non-dominated front
                parent_idx = nd_indices[rng.randint(len(nd_indices))]

                # Select victim
                dom_mask = ~nd_mask
                dom_indices = np.where(dom_mask)[0]
                if len(dom_indices) > 0:
                    # Remove worst dominated point
                    victim_idx = dom_indices[int(np.argmax(np.sum(F[dom_indices], axis=1)))]
                else:
                    # All non-dominated: remove point with lowest HV contribution
                    contribs = self._hv_contributions(F, ref)
                    nd_contribs = contribs[nd_indices]
                    min_pos = np.argmin(nd_contribs)
                    victim_idx = nd_indices[min_pos]

            parent = X[parent_idx].copy()

            # Adaptive mutation: start wide, narrow over time
            progress = used / self.budget
            step = (0.35 - 0.30 * progress) * (hi - lo)

            # Gaussian mutation
            child = parent + step * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = eval_point(child)
            if fc is None:
                break

            # Replace victim in population
            X[victim_idx] = child
            F[victim_idx] = fc

        return all_X

# EVOLVE-BLOCK-END
