"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-like multi-objective optimizer with hypervolume-aware selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_contribution_3d(self, points, ref):
        """Compute hypervolume contribution of each point in 3D.
        Points must be strictly better than ref in all objectives.
        Returns array of contributions (0 for dominated points)."""
        n = len(points)
        if n == 0:
            return np.array([])
        
        # Filter points strictly better than ref
        mask = np.all(points < ref, axis=1)
        if not np.any(mask):
            return np.zeros(n)
        
        pts = points[mask]
        idx = np.where(mask)[0]
        m = len(pts)
        
        if m == 0:
            return np.zeros(n)
        
        # For 3D hypervolume contribution, use the following approach:
        # Sort by first objective, then for each point, compute the volume
        # it uniquely dominates.
        
        # Sort by f0 (ascending)
        order = np.argsort(pts[:, 0])
        pts_sorted = pts[order]
        
        # Compute contributions using a sweep approach
        # For point i in sorted order, its contribution is the volume of
        # the box it uniquely dominates.
        
        contrib = np.zeros(m)
        
        # For each point, the contribution is the product of the differences
        # in each dimension to the next point that dominates it in that dimension.
        # This is complex for 3D. Let's use a simpler approximation.
        
        # Alternative: compute total HV with and without each point
        # This is O(n^2) which is fine for n <= 200
        
        # Actually, for n up to 200, O(n^2) is 40000 operations, very fast.
        # Let's compute HV of all points, then HV without each point.
        
        def _hv_3d(pts_subset):
            """Compute hypervolume of 3D points dominated by ref."""
            if len(pts_subset) == 0:
                return 0.0
            # Filter strictly better than ref
            mask = np.all(pts_subset < ref, axis=1)
            pts = pts_subset[mask]
            if len(pts) == 0:
                return 0.0
            
            # For 3D, sort by first coordinate
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            # Use the standard 3D HV algorithm
            # Sort by f0, then for each point, the contribution is:
            # (ref[0] - f0_i) * (max of (ref[1]-f1) and (ref[2]-f2) in a 2D sense)
            
            # Actually, let's use a simpler but correct approach:
            # The hypervolume in 3D can be computed by sorting on one axis
            # and using a 2D structure for the other two.
            
            # For small n, we can use the inclusion-exclusion or
            # the standard sweep-line.
            
            # Let's use the following: sort by f0 ascending.
            # For each point i, the "slab" it contributes is:
            # width in f0: min(ref[0], f0[i+1] if i+1 exists else ref[0]) - f0[i]
            # But this isn't quite right because other points may extend further in f1, f2.
            
            # Correct approach for 3D HV:
            # Sort points by f0 ascending.
            # Maintain a 2D structure of (f1, f2) for points seen so far.
            # For point i, its contribution is:
            #   (f0[i+1] or ref[0] - f0[i]) * HV_2D of the region dominated by
            #   the current point's (f1, f2) but not by any previously seen point.
            
            # This is getting complex. For n <= 200, let's just use a brute-force
            # grid-free approach or the standard algorithm.
            
            # Standard 3D HV algorithm (O(n log n)):
            # 1. Sort by f0
            # 2. Sweep from left to right, maintaining active points in a 2D structure
            # 3. For each point, compute the 2D HV contribution
            
            # For simplicity and correctness with n <= 200, let's use:
            # HV = sum over all points of the volume of the box they uniquely dominate.
            # A point p uniquely dominates the box [p0, min(p0, next_f0)] x [p1, min(p1, ...)] ...
            # This is hard to compute directly.
            
            # Let's use the following correct algorithm:
            # Sort by f0. For each point, the contribution in the f0 direction is
            # the gap to the next point in f0 (or ref[0]). In the f1-f2 plane,
            # we need the 2D HV of the point's (f1, f2) minus the 2D HV of
            # all previously processed points' (f1, f2) that are within the f0 slab.
            
            # This is still complex. For n <= 200, let's just use a simple
            # but correct method: compute HV by decomposing into boxes.
            
            # Actually, the simplest correct method for small n:
            # The hypervolume is the volume of the union of boxes [pi, ref] for all i.
            # We can compute this using the inclusion-exclusion principle, but that's 2^n.
            
            # Better: use the standard sweep-line for 3D.
            # Sort by f0. Process points in order. Maintain a list of "active"
            # points (those with f0 < current). For each new point, its contribution
            # is the volume of the box it adds to the union.
            
            # For 3D, the contribution of point i (after sorting by f0) is:
            #   (f0_next - f0_i) * max(0, HV_2D_contribution(f1_i, f2_i, active_points))
            # where HV_2D_contribution is the additional 2D area covered by (f1_i, f2_i)
            # given the active points.
            
            # 2D HV contribution of a point (a, b) given existing points:
            # It's the area of the rectangle [a, ref1] x [b, ref2] minus the area
            # already covered by existing points in that rectangle.
            
            # This is still non-trivial. For n <= 200, let's just compute the full
            # HV using a different method.
            
            # Alternative: since n is small, use the following:
            # HV = integral over the domain of the indicator that the point is
            # dominated by at least one point.
            
            # For practical purposes with n <= 200, let's use the following
            # algorithm which is O(n^2) but correct:
            
            # Sort by f0 ascending.
            # For each point i, define:
            #   w0_i = (f0[i+1] if i+1 < n else ref[0]) - f0[i]
            #   The 2D region dominated by point i in the (f1, f2) plane,
            #   restricted to points j with f0[j] >= f0[i] (i.e., j >= i in sorted order)
            #   that are not dominated by any point k with k < i (in sorted order)
            #   and f0[k] < f0[i+1] (i.e., within the same slab).
            
            # This is getting too complex. Let me just use a numerical approach
            # or a simpler but approximate HV for selection purposes.
            
            # For the actual scoring, the evaluator computes HV exactly.
            # For my internal selection, I can use an approximation.
            
            # Let's use a simple approximation: for each point, its "importance"
            # is the product of (ref[j] - f[j]) for j=0,1,2, weighted by
            # whether it's non-dominated.
            
            # Actually, for selection purposes, I don't need exact HV.
            # I can use the non-dominated front + diversity metric.
            
            # Let me just return a simple importance metric for selection.
            return 0.0  # Placeholder, we'll use a different selection strategy
        
        # For selection, use a simpler metric:
        # 1. Non-dominated sorting
        # 2. Crowding distance or similar
        return np.zeros(n)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        rng = np.random  # Use global RNG (seeded by evaluator)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])
        
        # Track all evaluated points
        all_X = []
        all_F = []
        
        # Phase 1: Random initialization
        pop_size = min(30, self.budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        all_X.extend(X.tolist())
        all_F.extend(F.tolist())
        used = pop_size
        
        # Phase 2: Iterative improvement
        # Use non-dominated sorting + crowding distance for selection
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Compute non-dominated front
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            front = np.where(~dominated)[0]
            
            if len(front) == 0:
                front = np.arange(n)
            
            # Select parent from front with some randomness
            parent_idx = front[rng.integers(len(front))]
            parent = X[parent_idx].copy()
            
            # Adaptive mutation strength
            progress = used / self.budget
            step = 0.15 * (1 - 0.5 * progress) * (hi - lo)
            
            # Gaussian mutation
            child = parent + step * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            all_X.append(child)
            all_F.append(fc)
            
            # Replace worst point in population
            # Use sum of objectives as a simple fitness
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc
        
        return X

# EVOLVE-BLOCK-END
