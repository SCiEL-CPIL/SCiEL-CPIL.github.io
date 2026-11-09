import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0 # 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1.1, 1.3, 1.1])
        
        # Initial Population
        pop_size = 20
        if self.budget < pop_size:
             pop_size = self.budget
             
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Calculate Hypervolume of a set of points (3D)
        def calc_hv(points, ref_point):
            # points: (N, 3)
            # ref_point: (3,)
            if len(points) == 0:
                return 0.0
            
            # Filter points that are better than ref in all objectives
            mask = np.all(points < ref_point, axis=1)
            if not np.any(mask):
                return 0.0
            pts = points[mask]
            
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            hv = 0.0
            # Sweep over f0
            # For each point i, it contributes a volume from pts[i,0] to pts[i+1,0] (or ref[0])
            # The cross section is the 2D HV of the points j such that pts[j,0] <= pts[i,0]
            # But wait, the standard sweep is:
            # Sort by f0.
            # Maintain a structure of the "skyline" in f1-f2 space?
            # For N=20, we can just do:
            # For each point p_i, the volume it uniquely contributes is complex.
            # Easier: 
            # HV = sum over all pairs? No.
            
            # Efficient 3D HV for small N:
            # 1. Sort by f0.
            # 2. Iterate i from 0 to N-1.
            #    The point p_i defines a box from p_i to ref.
            #    But we must subtract overlaps.
            #    Standard algorithm:
            #    Sort by f0.
            #    hv = 0
            #    for i in range(N):
            #       # The segment in f0 is from p_i[0] to (p_{i+1}[0] if i<N-1 else ref[0])
            #       # The cross section at this f0 slice is the 2D HV of points j where p_j[0] <= current_f0_end
            #       # Actually, simpler:
            #       # The volume contributed by p_i in the f0-slice [p_i[0], next_f0] is:
            #       # (next_f0 - p_i[0]) * HV_2D({p_j[1:3] for j <= i} minus those dominated by others in 2D?)
            
            # Let's use a simpler, robust O(N^3) or O(N^2 log N) approach for N=20.
            # Since N is very small, we can compute the union of boxes.
            # Box for point p: [p[0], ref[0]] x [p[1], ref[1]] x [p[2], ref[2]]
            # We need the volume of the union of these boxes.
            
            # Algorithm for Union of Boxes in 3D (small N):
            # Sort by x.
            # Sweep x.
            # Maintain a list of active intervals in y-z plane?
            # For N=20, we can just enumerate all "corners" or use the inclusion-exclusion? No, too complex.
            
            # Let's stick to the standard sweep:
            # Sort points by f0.
            # hv = 0
            # prev_f0 = ref[0] # No, we go from low f0 to high f0.
            # Actually, the volume is bounded by ref.
            # Points with f0 < ref[0] are valid.
            
            # Let's sort by f0 ascending.
            # p[0] < p[1] < ...
            # The hypervolume is the sum of volumes of boxes:
            # Box i: [p_i[0], p_{i+1}[0]] x [min_f1, ref[1]] x [min_f2, ref[2]]? No.
            
            # Correct Sweep:
            # Sort by f0.
            # hv = 0
            # for i in range(N):
            #    f0_end = pts[i+1, 0] if i < N-1 else ref[0]
            #    if f0_end <= pts[i, 0]: continue
            #    # The cross section at any f0 in [pts[i,0], f0_end] is the 2D HV of the set of points {p_j[1:3] | j <= i}
            #    # Let S_i = {p_0[1:3], ..., p_i[1:3]}
            #    # We need HV_2D(S_i, ref[1:3])
            #    # But we only need the part where f1 < ref[1] and f2 < ref[2].
            #    # Also, the 2D HV calculation must account for the fact that we are looking at the "skyline" in 2D.
            
            # Let's implement a helper for 2D HV.
            def hv_2d(pts_2d, ref_2d):
                # pts_2d: (M, 2), ref_2d: (2,)
                if len(pts_2d) == 0: return 0.0
                # Filter
                mask = np.all(pts_2d < ref_2d, axis=1)
                if not np.any(mask): return 0.0
                p = pts_2d[mask]
                # Sort by first coord
                p = p[np.argsort(p[:, 0])]
                hv = 0
                # Sweep f1
                # The volume is sum over segments in f1 of (width * height)
                # Height is determined by the min f2 of points active in that segment?
                # No, in 2D, HV is the area under the skyline.
                # Sort by f1.
                # Maintain min_f2? No.
                # The skyline in 2D is a set of points where no point dominates another.
                # Find non-dominated points in 2D.
                # Sort by f1 asc.
                # Iterate, keep track of min_f2 seen so far.
                # If p[i][1] < min_f2, it's on the skyline.
                # The area contributed by this point is (p[i][1] - next_f1_on_skyline) * (ref[1] - p[i][1])? No.
                # Let's just compute the union of rectangles.
                # Rect i: [p_i[0], ref[0]] x [p_i[1], ref[1]]
                # Sort by p_i[0].
                # Sweep x.
                # Keep track of the "active" minimum y?
                # Actually, for 2D, if we sort by x, the skyline is formed by points with decreasing y.
                # Find non-dominated set in 2D.
                # Sort by x asc.
                # nd = []
                # min_y = ref[1]
                # for p in sorted_pts:
                #    if p[1] < min_y:
                #       nd.append(p)
                #       min_y = p[1]
                # Now we have the skyline points sorted by x asc, y desc.
                # The HV is the sum of rectangles:
                # For each point p_i in nd (except last), width = nd[i+1][0] - nd[i][0]
                # For the last point, width = ref[0] - nd[-1][0]
                # Height for all is ref[1] - p_i[1]
                # Wait, the height is constant for the segment?
                # Yes, because the skyline is a step function.
                # Area = sum( (x_next - x_curr) * (ref_y - y_curr) )
                
                # Re-doing 2D HV properly:
                if len(p) == 1:
                    return (ref_2d[0] - p[0,0]) * (ref_2d[1] - p[0,1])
                
                # Find skyline
                # Sort by x
                p = p[np.argsort(p[:, 0])]
                skyline = []
                min_y = ref_2d[1]
                for i in range(len(p)):
                    if p[i, 1] < min_y:
                        skyline.append(p[i])
                        min_y = p[i, 1]
                        
                hv = 0
                for i in range(len(skyline)):
                    x_curr = skyline[i][0]
                    y_curr = skyline[i][1]
                    if i < len(skyline) - 1:
                        x_next = skyline[i+1][0]
                    else:
                        x_next = ref_2d[0]
                    width = x_next - x_curr
                    height = ref_2d[1] - y_curr
                    hv += width * height
                return hv

            # Back to 3D
            # Sort pts by f0
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0
            # We need to calculate the 2D HV of the set of points {p_0, ..., p_i} for each i?
            # This is O(N^3) if we do it naively. N=20, N^3=8000. Very fast.
            for i in range(len(pts)):
                f0_start = pts[i, 0]
                if i < len(pts) - 1:
                    f0_end = pts[i+1, 0]
                else:
                    f0_end = ref_point[0]
                
                if f0_end <= f0_start:
                    continue
                    
                # The cross section is the 2D HV of the points p_0..p_i projected to f1,f2
                # But we only care about points that are "active" in this slice.
                # All points p_j where j <= i are active in the slice [f0_start, f0_end]?
                # Yes, because they have f0 <= f0_start (since sorted).
                # So we compute HV_2D of {p_j[1:3] for j in 0..i}
                sub_pts = pts[:i+1, 1:3]
                hv_2 = hv_2d(sub_pts, ref_point[1:3])
                hv += (f0_end - f0_start) * hv_2
                
            return hv

        def get_marginal_hv_contributions(X_pop, F_pop):
            # Returns array of marginal HV for each point in X_pop
            # HV(S) is the HV of the set F_pop
            # HV(S \ {i}) is the HV of the set without point i
            # Contribution = HV(S) - HV(S \ {i})
            
            base_hv = calc_hv(F_pop, ref)
            contribs = np.zeros(len(F_pop))
            for i in range(len(F_pop)):
                # Create mask to exclude i
                mask = np.ones(len(F_pop), dtype=bool)
                mask[i] = False
                F_reduced = F_pop[mask]
                hv_reduced = calc_hv(F_reduced, ref)
                contribs[i] = base_hv - hv_reduced
            return contribs

        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Calculate contributions
            contribs = get_marginal_hv_contributions(X, F)
            
            # Selection
            # Parent 1: Best contribution (Exploitation)
            # Parent 2: Random (Exploration)
            # Or: Select 2 parents with probability proportional to contribution?
            # Let's try: One parent is the one with max contribution.
            # The other is random.
            p1_idx = np.argmax(contribs)
            p2_idx = rng.integers(0, len(X))
            
            # Crossover
            # Simple arithmetic crossover
            # child = 0.5 * (X[p1] + X[p2]) + noise
            child = 0.5 * (X[p1_idx] + X[p2_idx])
            
            # Mutation
            # Adaptive mutation? Start with 0.1, decrease?
            # Or keep constant 0.1-0.2
            sigma = 0.2
            child = child + sigma * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate
            f_child = problem(child)
            used += 1
            
            # Replacement
            # Find the index of the point with the MINIMUM marginal contribution
            # If the new point is worse than the worst point, replace it.
            # Actually, just replace the worst point.
            worst_idx = np.argmin(contribs)
            
            # Check if child is better than worst?
            # If child dominates worst, definitely replace.
            # If worst dominates child, maybe don't?
            # But we want to maximize HV.
            # Replacing the point with the lowest marginal HV with a new point
            # will increase or maintain HV if the new point adds unique volume.
            # If the new point is dominated by the rest, its marginal HV is 0.
            # The worst point has contribution >= 0.
            # If contribution(worst) > 0, and contribution(child) = 0, replacing decreases HV.
            # So we should only replace if contribution(child) > contribution(worst)?
            # Or simply: Calculate HV with child replacing worst. If > HV_current, replace.
            
            # Let's do the safe check:
            # Temporarily replace
            F_temp = F.copy()
            F_temp[worst_idx] = f_child
            X_temp = X.copy()
            X_temp[worst_idx] = child
            
            hv_new = calc_hv(F_temp, ref)
            hv_old = calc_hv(F, ref) # We have base_hv from contribs calc? 
            # Wait, base_hv was calculated at the start of the loop.
            # contribs[i] = base_hv - hv_without_i.
            # So base_hv is the current HV.
            
            if hv_new >= hv_old:
                F[worst_idx] = f_child
                X[worst_idx] = child
            else:
                # If it doesn't improve, we might still want to keep it for diversity?
                # No, for HV maximization, we should only accept if it improves or maintains.
                # But if we never replace, we might get stuck.
                # However, with random exploration, eventually a better point will be found.
                # To ensure progress, we can force replacement if the child is non-dominated
                # by the population?
                # Let's stick to HV improvement.
                pass

        return X
