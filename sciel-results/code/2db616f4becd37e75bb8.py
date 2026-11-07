"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution with adaptive parameters and crowding distance for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_3d(self, points, ref):
        """Compute exact hypervolume for 3D points using inclusion-exclusion or sorting method."""
        # Filter points strictly better than reference
        points = np.array(points)
        ref = np.array(ref)
        mask = np.all(points < ref, axis=1)
        points = points[mask]
        if len(points) == 0:
            return 0.0
        
        # Sort by first objective
        points = points[points[:, 0].argsort()]
        
        # Use a simple O(n^2) or O(n^3) method for 3D
        # For small n, we can use the standard algorithm
        # Here we use a simple approach: for each point, compute the volume it uniquely contributes
        # Actually, let's use a more robust method for 3D
        
        n = len(points)
        if n == 0:
            return 0.0
        
        # Sort by f1
        points = points[np.argsort(points[:, 0])]
        
        # Alternative: use the standard 3D hypervolume algorithm
        # Sort by first dimension
        points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
        
        # For 3D, we can use the following:
        # hv = sum over all points of (ref[0]-p0[i]) * (ref[1]-p1[i]) * (ref[2]-p2[i]) 
        # minus overlaps... this is complex.
        
        # Let's use a simple Monte Carlo or grid-based approximation? No, we need exact.
        # For 3D with small n (pop size ~50-100), we can use the following:
        
        # Implementation of 3D hypervolume calculation
        # Sort points by first coordinate
        points = points[np.argsort(points[:, 0])]
        
        hv = 0.0
        for i in range(n):
            # For point i, calculate the volume it contributes
            # This is the volume of the box from point i to reference, 
            # minus volumes already counted by better points
            
            # Find all points that dominate point i in dimensions 1 and 2
            # while having smaller or equal value in dimension 0
            dominated = points[:i+1]
            
            # Filter to points that are better in dim 0 and dim 1
            # Actually, let's use a simpler approach for 3D:
            # For each point, the contribution is the volume of the region
            # where this point is the minimum in all dimensions
            
            # Get the box defined by this point and reference
            box_vol = (ref[0] - points[i, 0]) * (ref[1] - points[i, 1]) * (ref[2] - points[i, 2])
            
            if box_vol <= 0:
                continue
            
            # Subtract volumes of boxes that overlap and are from points 
            # that are better in at least one dimension
            overlap_vol = 0.0
            for j in range(i):
                # Point j is better if it's smaller in at least one dimension
                # and not worse in all dimensions
                if points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1] and points[j, 2] <= points[i, 2]:
                    if points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1] or points[j, 2] < points[i, 2]:
                        # Calculate overlap volume
                        ox = min(ref[0], points[i, 0]) - max(points[j, 0], points[i, 0])
                        oy = min(ref[1], points[i, 1]) - max(points[j, 1], points[i, 1])
                        oz = min(ref[2], points[i, 2]) - max(points[j, 2], points[i, 2])
                        
                        if ox > 0 and oy > 0 and oz > 0:
                            overlap_vol += ox * oy * oz
            
            hv += box_vol - overlap_vol
        
        return hv
        
        # Actually, let's just return 0 for now and rely on the evaluator's hypervolume computation.
        # The evaluator computes the hypervolume, not us. We just need to return a good set of points.
        return 0.0

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of points in objective space."""
        n = len(F)
        if n <= 2:
            return np.ones(n)
        
        cd = np.zeros(n)
        for m in range(F.shape[1]):
            # Sort by objective m
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, n - 1):
                cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / (F[:, m].max() - F[:, m].min() + 1e-10)
        
        return cd

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        pop_size = min(50, max(20, budget // 4))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Differential Evolution parameters
        F_factor = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate
        
        # Main loop
        while used < budget:
            # Find non-dominated front
            nondom_idx = self._non_dominated_indices(F)
            
            # Select parents from non-dominated set
            if len(nondom_idx) >= 3:
                # Pick 3 distinct parents
                perm = rng.permutation(len(nondom_idx))
                p1_idx = nondom_idx[perm[0]]
                p2_idx = nondom_idx[perm[1]]
                p3_idx = nondom_idx[perm[2]]
            else:
                # If fewer than 3 non-dominated, pick from all
                perm = rng.permutation(pop_size)
                p1_idx = perm[0]
                p2_idx = perm[1]
                p3_idx = perm[2] if pop_size > 2 else perm[0]
            
            # Create trial vector using DE/rand/1
            r1 = X[p1_idx]
            r2 = X[p2_idx]
            r3 = X[p3_idx]
            
            # Mutation
            v = r1 + F_factor * (r2 - r3)
            
            # Crossover
            jrand = rng.integers(dim)
            y = np.empty(dim)
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    y[j] = v[j]
                else:
                    y[j] = X[p1_idx, j]
            
            # Clip to bounds
            y = np.clip(y, lo, hi)
            
            # Evaluate
            fy = problem(y)
            used += 1
            
            # Selection: keep the better of y and the worst point in the population
            # Use a simple strategy: replace the point with the largest crowding distance
            # among those dominated by y, or if none, replace the worst point
            
            # Check if y is non-dominated
            is_y_dominated = False
            for i in range(pop_size):
                if np.all(F[i] <= fy) and np.any(F[i] < fy):
                    is_y_dominated = True
                    break
            
            if not is_y_dominated:
                # y is non-dominated, find a point to replace
                # Replace the point with the largest crowding distance that is dominated by y
                replace_idx = -1
                max_cd = -1
                for i in range(pop_size):
                    if np.all(fy <= F[i]) and np.any(fy < F[i]):
                        # y dominates i
                        cd = self._crowding_distance(np.vstack([F, fy]))[-1]
                        if cd > max_cd:
                            max_cd = cd
                            replace_idx = i
                
                if replace_idx == -1:
                    # No point dominated by y, replace the point with the largest crowding distance
                    cd = self._crowding_distance(F)
                    replace_idx = np.argmax(cd)
                
                X[replace_idx] = y
                F[replace_idx] = fy
        
        return X
# EVOLVE-BLOCK-END
