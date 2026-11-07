"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using Sobol sampling + adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _sobol_sample(self, n, dim, rng):
        """Generate n quasi-random points in [0,1]^dim using a simple Sobol-like sequence."""
        # Use numpy's random generator with a low-discrepancy approach
        # For simplicity, use scrambled Sobol via inverse CDF of uniform
        # Actually, let's use a stratified sampling approach that approximates Sobol
        # Divide [0,1] into n^dim/n = n^(dim-1) cells per dimension... 
        # Simpler: use the van der Corput sequence generalized to dim dimensions
        points = np.zeros((n, dim))
        for d in range(dim):
            # van der Corput sequence in base 2
            for i in range(n):
                val = 0.0
                denom = 2.0
                x = i
                while x > 0:
                    val += (x % 2) / denom
                    x //= 2
                    denom *= 2.0
                points[i, d] = val
        return points

    def _dtlz4_transform(self, x, dim):
        """
        Transform a point in [0,1]^dim to DTLZ4 space.
        DTLZ4: first dim-2 variables are 'position' variables, last 2 are 'distance' variables.
        g(x) = 100 * (sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)) for i in dim-2..dim-1) + (dim-2))
        f1 = (1+g)*x_1*...*x_{dim-2}
        f2 = (1+g)*(1-x_1)*x_2*...*x_{dim-2}
        f3 = (1+g)*(1-x_{dim-2})
        """
        n_pos = dim - 2
        pos_vars = x[:n_pos]
        dist_vars = x[n_pos:]
        
        # DTLZ4 distance transformation: x_i^0.1 for distance variables
        dist_transformed = dist_vars ** 0.1
        
        # g function
        g = 100.0 * (np.sum((dist_transformed - 0.5)**2 - np.cos(20 * np.pi * (dist_transformed - 0.5))) + n_pos)
        
        # Objective functions
        f1 = (1 + g) * np.prod(pos_vars)
        f2 = (1 + g) * (1 - pos_vars[0]) * np.prod(pos_vars[1:]) if n_pos > 1 else (1 + g) * (1 - pos_vars[0])
        f3 = (1 + g) * (1 - pos_vars[-1])
        
        return np.array([f1, f2, f3])

    def _hypervolume_3d(self, points, ref):
        """Compute 3D hypervolume for points strictly better than ref."""
        # Filter points strictly better than reference
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        hv = 0.0
        prev_y = ref[1]
        prev_z = ref[2]
        
        for i in range(len(pts)):
            x, y, z = pts[i]
            # Volume contribution: (x_prev - x) * (y_prev - y) * (z_prev - z)
            # But we need to be careful with the sweep line approach
            
        # Use a simpler approach: for 3D, use the standard algorithm
        # Sort by f1, then for each point, compute the 2D hypervolume of the remaining points
        # that are dominated by this point in f2 and f3
        
        # Actually, let's use a cleaner 3D HV computation
        # Sort by first objective ascending
        pts_sorted = pts[np.argsort(pts[:, 0])]
        
        hv = 0.0
        # Maintain a list of (y, z) pairs that are non-dominated in the 2D sense
        # For each new point, remove any (y,z) that it dominates
        
        active = []  # list of [y, z]
        
        for i in range(len(pts_sorted)):
            x, y, z = pts_sorted[i]
            
            # Remove any active points that are dominated by (y, z)
            new_active = []
            for ay, az in active:
                if not (y <= ay and z <= az and (y < ay or z < az)):
                    new_active.append([ay, az])
            active = new_active
            active.append([y, z])
            
            # Compute the 2D hypervolume of active points with reference (ref[1], ref[2])
            # and multiply by the x-slice width
            # The x-slice for this point is from pts_sorted[i][0] to pts_sorted[i+1][0] (or ref[0])
            
        # This is getting complex. Let me use a simpler but correct approach.
        # For small n, we can use the inclusion-exclusion or a grid-based approximation.
        # But since we need exact HV, let's use the standard 3D algorithm.
        
        # Alternative: use the fact that for the Pareto front of DTLZ4, we know the shape.
        # But we need to compute HV of arbitrary recorded points.
        
        # Let me implement a proper 3D HV using the "skyline" approach.
        # Sort by f1 ascending.
        # For each point i, the contribution to HV is:
        #   (x_{i-1} - x_i) * HV_2D({(y_j, z_j) : j >= i, y_j < y_i, z_j < z_i})
        # where x_{i-1} is the f1 of the previous point (or ref[0] for i=0).
        
        # Actually, the standard approach:
        # Sort points by f1 ascending.
        # For point i, its "slice" in f1 is [f1_i, f1_{i+1}) where f1_{i+1} is the next point's f1 or ref[0].
        # Within this slice, the 2D region that is dominated by point i but not by any earlier point
        # is the 2D HV of the set of (f2, f3) of points j >= i, restricted to the rectangle
        # [0, f2_i) x [0, f3_i) minus the parts already covered by points j < i.
        
        # This is still complex. Let me just use a simple but potentially slower method for small n.
        
        # For n < 500, we can use the following:
        # HV = sum over all subsets... no, that's exponential.
        
        # Let me use the "divide and conquer" or just the standard sweep.
        
        # I'll implement the standard 3D HV algorithm from the literature.
        
        # Sort by first coordinate
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        
        n = len(pts)
        if n == 0:
            return 0.0
        
        # Build the skyline in 2D for the remaining dimensions
        # Process points in order of increasing f1
        # Maintain a list of non-dominated (f2, f3) pairs
        
        # For each point, compute the 2D HV of the region it uniquely dominates
        # in the slice [f1_i, f1_{i+1}]
        
        hv = 0.0
        # active: list of (f2, f3) for points processed so far that are still relevant
        # We need to compute, for each point i, the 2D area of the rectangle [0, f2_i] x [0, f3_i]
        # that is not covered by any previously processed point's rectangle.
        
        # This is equivalent to: 2D HV of the set of (f2, f3) of points j >= i,
        # but restricted to points that are not dominated by any point j' < i.
        
        # Let me just compute it directly for each slice.
        
        # For efficiency, I'll maintain the set of non-dominated (f2, f3) points.
        # When adding a new point (f2, f3), remove any point that it dominates.
        
        skyline = []  # list of [f2, f3]
        
        for i in range(n):
            f1, f2, f3 = pts[i]
            f1_next = pts[i+1][0] if i + 1 < n else ref[0]
            
            # Remove from skyline any point dominated by (f2, f3)
            new_skyline = []
            for sf2, sf3 in skyline:
                if not (f2 <= sf2 and f3 <= sf3 and (f2 < sf2 or f3 < sf3)):
                    new_skyline.append([sf2, sf3])
            skyline = new_skyline
            skyline.append([f2, f3])
            
            # Compute 2D HV of skyline with reference (ref[1], ref[2])
            # 2D HV: sort by f2, then sweep
            skyline_sorted = sorted(skyline, key=lambda p: p[0])
            
            area_2d = 0.0
            prev_f3 = ref[2]
            for j in range(len(skyline_sorted)):
                sf2, sf3 = skyline_sorted[j]
                if sf2 < ref[1] and sf3 < ref[2]:
                    # The contribution is (sf2 - prev_f2) * (prev_f3 - sf3)
                    # Wait, we need to be more careful.
                    # Sort by f2 ascending. For each point, the width in f2 is from the previous point's f2 (or 0) to this point's f2.
                    # The height in f3 is from this point's f3 to the previous point's f3 (or ref[2]).
                    pass
            
            # Let me compute 2D HV properly
            # Sort by f2 ascending
            skyline_sorted = sorted(skyline, key=lambda p: p[0])
            
            area_2d = 0.0
            prev_x = 0.0
            for j in range(len(skyline_sorted)):
                sf2, sf3 = skyline_sorted[j]
                if sf2 >= ref[1]:
                    break
                # The x-slice is [prev_x, sf2]
                # The y-range is [0, min(sf3, ref[2])]
                # But we need to account for the fact that previous points may have higher f3
                # Actually, since skyline is non-dominated, f3 decreases as f2 increases
                # So the height at f2 is sf3
                width = sf2 - prev_x
                height = min(sf3, ref[2])
                area_2d += width * height
                prev_x = sf2
            
            # Add the final slice from last f2 to ref[1]
            if prev_x < ref[1]:
                # The height is 0 here since no point covers this region
                pass
            
            hv += (f1_next - f1) * area_2d
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use Sobol-like quasi-random sampling for initial population
        n_init = min(50, self.budget // 4)
        
        # Generate quasi-random points
        X = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            for i in range(n_init):
                val = 0.0
                denom = 2.0
                x = i
                while x > 0:
                    val += (x % 2) / denom
                    x //= 2
                    denom *= 2.0
                X[i, d] = val
        
        # Map to bounds
        X = lo + (hi - lo) * X
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Keep track of all evaluated points
        all_X = X.copy()
        all_F = F.copy()
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            if np.any(~dominated):
                parents_idx = np.where(~dominated)[0]
            else:
                parents_idx = np.arange(len(F))
            
            # Select a parent
            p_idx = parents_idx[rng.integers(len(parents_idx))]
            parent = X[p_idx]
            
            # Adaptive mutation
            # Use a smaller step size for later in the budget
            progress = used / self.budget
            step_size = 0.1 * (1.0 - 0.5 * progress)
            
            child = parent + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace the worst point in the population
            # Use a simple crowding distance or just replace the point with highest sum
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
            
            # Add to all evaluated points
            all_X = np.vstack([all_X, child])
            all_F = np.vstack([all_F, fc])
        
        return all_X
# EVOLVE-BLOCK-END
