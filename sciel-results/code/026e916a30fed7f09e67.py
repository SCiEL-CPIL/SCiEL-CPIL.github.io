"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using hypervolume-aware selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_contribution_3d(self, points, ref):
        """
        Calculate the hypervolume contribution of each point in a 3D set.
        points: (N, 3) array of non-dominated points.
        ref: (3,) array of reference point.
        Returns: (N,) array of HV contributions.
        """
        n = len(points)
        if n == 0:
            return np.array([])
        
        # Filter strictly better than ref
        mask = np.all(points < ref, axis=1)
        if not np.any(mask):
            return np.zeros(n)
        
        pts = points[mask]
        idx = np.where(mask)[0]
        m = len(pts)
        
        contrib = np.zeros(n)
        
        # Exact method: for each point, compute HV of full set minus HV without it.
        # For N=20, this is 20 * HV_calc. HV_calc via sweep is fast.
        
        def _calc_hv_3d(pts_subset):
            if len(pts_subset) == 0:
                return 0.0
            msk = np.all(pts_subset < ref, axis=1)
            p = pts_subset[msk]
            if len(p) == 0:
                return 0.0
            
            # Sort by f0
            p = p[np.argsort(p[:, 0])]
            n_p = len(p)
            hv = 0.0
            
            # Sweep line: maintain list of active "fronts" in f1-f2 plane
            # Actually, a simpler O(N^2) for small N:
            # For each point i in sorted order, its contribution is:
            # (f0[i+1] - f0[i]) * area_of_2d_front(points[i+1:])
            # But this is not quite right because points[i] itself contributes to the area.
            # Standard algorithm:
            # Sort by f0 ascending.
            # For i from 0 to N-2:
            #   width = min(f0[i+1], ref[0]) - f0[i]  (actually just f0[i+1]-f0[i] if sorted)
            #   area = HV_2D of points[i:] projected to f1,f2 with ref
            #   hv += width * area
            # hv += (ref[0] - f0[N-1]) * HV_2D(points[N-1:])
            
            # HV_2D: given points (M,2), ref_2 (2,), compute HV.
            # Sort by f1. For each point, contribution is (f1[i+1]-f1[i])*(ref2[1]-f2[i])
            # where f1[i+1] is next point's f1 or ref.
            
            def hv_2d(sub_pts, ref_2):
                if len(sub_pts) == 0:
                    return 0.0
                # Filter
                m2 = np.all(sub_pts < ref_2, axis=1)
                sp = sub_pts[m2]
                if len(sp) == 0:
                    return 0.0
                sp = sp[np.argsort(sp[:, 0])]
                n_s = len(sp)
                vol = 0.0
                for i in range(n_s):
                    next_x = sp[i+1, 0] if i + 1 < n_s else ref_2[0]
                    width = next_x - sp[i, 0]
                    height = ref_2[1] - sp[i, 1]
                    vol += width * height
                return vol
            
            # Now do 3D
            # Sort p by f0
            # p is already sorted
            # We need to be careful: the "front" at position i is formed by all points j >= i
            # but we want the 2D HV of the projection of the Pareto front of {p[i], p[i+1], ...}
            # Actually, the standard sweep adds contributions incrementally.
            
            # Let's just use the removal method since N is small.
            # HV(full) - HV(full \ {i})
            # This is O(N^3) total if HV is O(N^2).
            # For N=20, 20 * (20^2) = 8000 ops per call. 200 calls = 1.6M ops. Fine.
            
            pass  # We will implement this in the outer loop instead
            
        # Better: compute HV of full set, then for each i, compute HV without i.
        # But we can optimize by just doing the grid method for the whole set once?
        # No, we need per-point contributions.
        
        # Let's use the efficient O(N^2) removal method with a fast HV calculator.
        
        def hv_2d_calc(sub_pts, ref_2):
            if len(sub_pts) == 0:
                return 0.0
            m2 = np.all(sub_pts < ref_2, axis=1)
            sp = sub_pts[m2]
            if len(sp) == 0:
                return 0.0
            sp = sp[np.argsort(sp[:, 0])]
            n_s = len(sp)
            vol = 0.0
            for i in range(n_s):
                next_x = sp[i+1, 0] if i + 1 < n_s else ref_2[0]
                width = next_x - sp[i, 0]
                height = ref_2[1] - sp[i, 1]
                vol += width * height
            return vol
        
        def hv_3d_calc(p):
            if len(p) == 0:
                return 0.0
            msk = np.all(p < ref, axis=1)
            pp = p[msk]
            if len(pp) == 0:
                return 0.0
            pp = pp[np.argsort(pp[:, 0])]
            n_p = len(pp)
            hv = 0.0
            for i in range(n_p):
                # The "slice" at f0[i] has width to the next point or ref
                next_f0 = pp[i+1, 0] if i + 1 < n_p else ref[0]
                width = next_f0 - pp[i, 0]
                # The area is the HV_2D of the Pareto front of pp[i:] in f1,f2
                # But we need the HV of the set pp[i:] projected to 2D.
                # This is exactly hv_2d_calc(pp[i:, 1:], ref[1:])
                area = hv_2d_calc(pp[i:, 1:], ref[1:])
                hv += width * area
            return hv
        
        full_hv = hv_3d_calc(pts)
        
        for k, idx_k in enumerate(idx):
            # Remove point k
            mask_remove = np.ones(m, dtype=bool)
            mask_remove[k] = False
            hv_without = hv_3d_calc(pts[mask_remove])
            contrib[idx_k] = full_hv - hv_without
        
        return contrib

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])
        
        pop_size = 20
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Precompute box volumes for quick checks
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Identify non-dominated points
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
                            
            nd_idx = np.where(~dominated)[0]
            
            if len(nd_idx) == 0:
                parent_idx = rng.integers(pop_size)
                victim_idx = int(np.argmax(np.sum(F, axis=1)))
            else:
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
                
                dom_idx = np.where(dominated)[0]
                if len(dom_idx) > 0:
                    victim_idx = dom_idx[int(np.argmax(np.sum(F[dom_idx], axis=1)))]
                else:
                    # Use HV contribution for selection
                    # Compute contributions for non-dominated points
                    contribs = self._hv_contribution_3d(F, ref)
                    # Find the non-dominated point with smallest contribution
                    # Initialize to -inf
                    min_contrib = float('inf')
                    victim_idx = nd_idx[0]
                    for idx in nd_idx:
                        if contribs[idx] < min_contrib:
                            min_contrib = contribs[idx]
                            victim_idx = idx
            
            # Generate child with adaptive mutation
            parent = X[parent_idx]
            # Adaptive scale: start larger, decay over time
            progress = used / self.budget
            mutation_scale = 0.3 * (1.0 - progress) + 0.05
            child = parent + mutation_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            X[victim_idx] = child
            F[victim_idx] = fc
            
        return X
# EVOLVE-BLOCK-END
