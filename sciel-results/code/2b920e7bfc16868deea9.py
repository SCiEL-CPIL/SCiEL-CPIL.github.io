# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_3d(self, points, ref):
        """
        Calculate the 3D hypervolume indicator for a set of points.
        points: list of arrays of shape (3,)
        ref: array of shape (3,)
        """
        if not points:
            return 0.0
        
        # Filter points that are strictly better than ref in all objectives
        # The problem statement says "Only points strictly better than the reference point in every objective count."
        # However, for internal selection, we can use all non-dominated points.
        # For the final score, the evaluator does the filtering.
        # Here we calculate HV of the given set.
        
        # Sort by first objective
        points = sorted(points, key=lambda p: p[0])
        
        # In 3D, we can project to 2D for each "layer" or use a sweep line.
        # Simpler: For each point, calculate the area of the rectangle in (f2, f3) 
        # that it "owns".
        
        # A point p_i contributes if it is not dominated by any previous point (in sorted order) 
        # in the (f2, f3) plane? No, it's more complex.
        
        # Let's use the standard algorithm for 3D HV:
        # Sort by f1.
        # Maintain a list of "active" points in the (f2, f3) plane.
        # Actually, for small N, we can do:
        # Total HV = sum over non-dominated points of contribution.
        
        # First, find the non-dominated subset
        nd_points = []
        for i in range(len(points)):
            dominated = False
            for j in range(len(points)):
                if i == j: continue
                if np.all(points[i] <= points[j]) and np.any(points[i] < points[j]):
                    dominated = True
                    break
            if not dominated:
                nd_points.append(points[i])
                
        if not nd_points:
            return 0.0
            
        # Sort non-dominated points by f1
        nd_points = sorted(nd_points, key=lambda p: p[0])
        
        total_hv = 0.0
        # We need to calculate the area in (f2, f3) for each point
        # that is not covered by points with smaller f1.
        
        # This is equivalent to finding the union of rectangles defined by 
        # (p_i[1], p_i[2]) and (ref[1], ref[2]) for all i, 
        # but weighted by (ref[0] - p_i[0]).
        
        # Let's use a simpler recursive or iterative method for 3D.
        # Since N is small (< 50), we can use the "slice" method.
        
        # Sort by f1
        # For each point i, its contribution is:
        # (ref[0] - p_i[0]) * Area( R_i \ U(R_j for j < i) )
        # where R_k is the rectangle [p_k[1], ref[1]] x [p_k[2], ref[2]]
        
        # To compute Area( R_i \ U(R_j for j < i) ), we can just compute the area of the union of rectangles 
        # for j < i, and subtract from area(R_i)? No, that's not quite right because the union is complex.
        
        # Alternative: 
        # The HV is the volume of the union of boxes B_i = [p_i[0], ref[0]] x [p_i[1], ref[1]] x [p_i[2], ref[2]].
        
        # Let's just use a standard 3D HV algorithm.
        # Reference: "A Simple and Efficient Algorithm for the 3-Dimensional Hypervolume Indicator"
        
        # 1. Sort points by f1
        # 2. For each point p_i, define the rectangle R_i in (f2, f3) space: 
        #    [p_i[1], ref[1]] x [p_i[2], ref[2]]
        # 3. The contribution of p_i is (ref[0] - p_i[0]) * Area(R_i \ U(R_j for j < i))
        
        # We need to compute the area of the difference of a rectangle and a union of rectangles.
        # This is doable with a sweep line or simple geometry since N is small.
        
        # Let's implement a helper to compute area of union of rectangles in 2D.
        def union_area_2d(rects):
            # rects: list of (y1, y2, z1, z2) where y1 < y2, z1 < z2
            # Returns area of union
            if not rects:
                return 0.0
            
            # Coordinate compression
            ys = set()
            zs = set()
            for r in rects:
                ys.add(r[0])
                ys.add(r[1])
                zs.add(r[2])
                zs.add(r[3])
                
            ys = sorted(list(ys))
            zs = sorted(list(zs))
            
            # Create a grid
            # Mark cells covered by any rectangle
            # Since N is small, the grid is small
            covered = np.zeros((len(ys)-1, len(zs)-1), dtype=bool)
            
            for r in rects:
                y1, y2, z1, z2 = r
                # Find indices
                i1 = ys.index(y1)
                i2 = ys.index(y2)
                j1 = zs.index(z1)
                j2 = zs.index(z2)
                
                covered[i1:i2, j1:j2] = True
            
            # Calculate area
            area = 0.0
            for i in range(len(ys)-1):
                for j in range(len(zs)-1):
                    if covered[i, j]:
                        area += (ys[i+1] - ys[i]) * (zs[j+1] - zs[j])
                        
            return area

        # Now compute HV
        prev_union = []
        total = 0.0
        for i in range(len(nd_points)):
            p = nd_points[i]
            # Rectangle for this point
            rect_i = (p[1], ref[1], p[2], ref[2])
            
            # Area of this rectangle
            area_i = (ref[1] - p[1]) * (ref[2] - p[2])
            
            # Area of union of previous rectangles
            if prev_union:
                # We need Area( R_i \ U_prev )
                # This is Area(R_i) - Area(R_i intersect U_prev)
                # It is easier to compute the union of prev_union + rect_i and subtract union of prev_union?
                # No, that's not efficient.
                
                # Let's just compute the union area of prev_union + rect_i
                # And subtract the union area of prev_union?
                # No, the contribution is (ref[0] - p[0]) * (Area(R_i) - Area(R_i \cap U_prev))
                # Area(R_i \cap U_prev) is the area of the part of R_i already covered.
                
                # Let's compute the union of all rectangles so far including current, 
                # and the union of all rectangles so far excluding current.
                # This is O(N^2) HV calculations. N=20, so 400 calcs. Each calc is O(N^2) grid. 16000 ops. Fine.
                
                pass # Placeholder
            
            # Actually, let's just compute the total HV of the set nd_points[:i+1] and subtract total HV of nd_points[:i]?
            # HV(S) is not additive like that because the "shadow" changes.
            
            # Let's stick to the definition:
            # Contribution of p_i = (ref[0] - p_i[0]) * Area(R_i \ U(R_j, j < i))
            
            # We need Area(R_i \ U_prev).
            # Let U_prev be the union of rectangles for j < i.
            # Area(R_i \ U_prev) = Area(R_i) - Area(R_i \cap U_prev).
            
            # To find Area(R_i \cap U_prev), we can intersect R_i with each rect in U_prev and find the union of those intersections.
            
            if i == 0:
                new_area = area_i
            else:
                # Find intersection of rect_i with each rect in prev_union
                # And compute the union area of those intersections
                intersections = []
                for r_prev in prev_union:
                    # Intersect rect_i (y1_i, y2_i, z1_i, z2_i) with r_prev (y1_p, y2_p, z1_p, z2_p)
                    y1 = max(p[1], r_prev[0])
                    y2 = min(ref[1], r_prev[1])
                    z1 = max(p[2], r_prev[2])
                    z2 = min(ref[2], r_prev[3])
                    
                    if y1 < y2 and z1 < z2:
                        intersections.append((y1, y2, z1, z2))
                        
                covered_area = union_area_2d(intersections)
                new_area = area_i - covered_area
                
            total += (ref[0] - p[0]) * new_area
            prev_union.append(rect_i)
            
        return total

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        pop_size = 20
        if budget < pop_size:
            pop_size = max(2, budget)
            
        # Step size
        # Initial step size should be small enough to stay in bounds initially
        widths = hi - lo
        min_width = np.min(widths)
        sigma = 0.5 * min_width
        
        # Mean
        mean = (lo + hi) / 2.0
        
        # Covariance
        C = np.eye(dim)
        
        # Paths for sigma adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Evaluation counter
        evals_used = 0
        
        # Storage for all evaluated points (for the evaluator)
        # The evaluator records what problem() returns. We just need to call it.
        
        # We need to select parents. We will keep the best mu points based on HV contribution.
        mu = pop_size // 2
        
        while evals_used < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            X = []
            for _ in range(pop_size):
                if evals_used >= budget:
                    break
                # Sample from normal
                z = np.random.randn(dim)
                x = mean + sigma * (C @ z) # Approximation, should be cholesky
                # Proper sampling: C = L L^T, x = mean + sigma * L z
                L = np.linalg.cholesky(C)
                x = mean + sigma * (L @ z)
                
                # Handle bounds
                # Reflection method
                for i in range(dim):
                    if x[i] < lo[i]:
                        x[i] = 2 * lo[i] - x[i]
                    elif x[i] > hi[i]:
                        x[i] = 2 * hi[i] - x[i]
                        
                # Clip to be safe
                x = np.clip(x, lo, hi)
                
                X.append(x)
                
            X = np.array(X)
            
            # Evaluate
            F = np.array([problem(x) for x in X])
            evals_used += len(X)
            
            # Selection
            # Find non-dominated points
            # And compute HV contribution for each
            # To save time, we can just use rank and crowding, or simple HV of the set.
            # Let's compute the HV contribution of each point in the non-dominated front.
            
            # First, get non-dominated indices
            nd_indices = []
            for i in range(len(F)):
                dominated = False
                for j in range(len(F)):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
                    
            if not nd_indices:
                # Fallback: select best by sum
                sum_f = np.sum(F, axis=1)
                nd_indices = np.argsort(sum_f)[:mu]
                
            # Compute HV contribution for each point in nd_indices
            # We need the reference point for internal selection.
            # The task ref is [3.0, 5.0, 7.0]. We can use this.
            ref_int = np.array([3.0, 5.0, 7.0])
            
            contributions = {}
            # We compute HV of the set, and HV of set minus each point.
            # This is O(N^2) HV calls.
            
            # Optimize: Just compute HV of the full set, and then for each point, 
            # compute HV of the set excluding it.
            # Since N is small, this is fine.
            
            # However, we can be smarter.
            # Let's just pick the points with the best "potential" using a simpler metric if HV is too slow.
            # But HV is the correct metric.
            
            # Let's try to compute contributions.
            # To avoid O(N^2) HV calls, we can use the fact that we only need to select mu points.
            # We can just compute the HV of the top candidates.
            
            # Let's just use the first mu non-dominated points if we are tight on time.
            # Or, let's compute the HV of the full set, and then subtract each point one by one?
            # No, that's not how it works.
            
            # Let's stick to a simpler selection: 
            # 1. Non-dominated sort.
            # 2. Crowding distance.
            # This is standard NSGA-II selection and is fast.
            
            # Compute crowding distance for nd_indices
            if len(nd_indices) <= 2:
                selected_indices = nd_indices
            else:
                # Sort by each objective
                crowd = {i: 0.0 for i in nd_indices}
                for j in range(n_obj):
                    sorted_idx = sorted(nd_indices, key=lambda i: F[i][j])
                    crowd[sorted_idx[0]] = float('inf')
                    crowd[sorted_idx[-1]] = float('inf')
                    f_min = F[sorted_idx[0]][j]
                    f_max = F[sorted_idx[-1]][j]
                    if f_max == f_min:
                        continue
                    for k in range(1, len(sorted_idx) - 1):
                        crowd[sorted_idx[k]] += (F[sorted_idx[k+1]][j] - F[sorted_idx[k-1]][j]) / (f_max - f_min)
                        
                # Select top mu by crowding
                selected_indices = sorted(nd_indices, key=lambda i: crowd[i], reverse=True)[:mu]
                
            # Update Mean
            weights = np.ones(mu) / mu
            mean = np.sum([weights[i] * X[idx] for i, idx in enumerate(selected_indices)], axis=0)
            
            # Update Covariance
            # C = sum w_i (x_i - mean) (x_i - mean)^T
            diffs = np.array([X[idx] - mean for idx in selected_indices])
            C = np.zeros((dim, dim))
            for i in range(mu):
                C += weights[i] * np.outer(diffs[i], diffs[i])
                
            # Ensure C is positive definite
            # Add small identity
            C += 1e-6 * np.eye(dim)
            
            # Update Sigma
            # Simple adaptation: if we found good points, keep sigma or decrease slightly
            # Standard CMA-ES uses path lengths.
            # Let's implement a simple version.
            
            # Compute path length in sigma space
            # p_sigma = (1 - c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * (mean_new - mean_old) / sigma
            # We need mean_old.
            
            # For simplicity and robustness in this short run, let's just adapt sigma based on the 
            # improvement in HV or just keep it constant/decreasing.
            # Given the short budget, a constant sigma or slowly decreasing sigma is often sufficient.
            # Let's decrease sigma by a factor of 0.95 each generation to ensure convergence.
            sigma *= 0.95
            
            # Ensure sigma doesn't become too small
            sigma = max(sigma, 1e-6 * min_width)

        return None
# EVOLVE-BLOCK-END
