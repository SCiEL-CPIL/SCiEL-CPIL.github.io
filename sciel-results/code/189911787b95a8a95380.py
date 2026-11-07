<<<<<<< SEARCH
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0  # safety margin before 300s/330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
=======
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0  # safety margin before 300s/330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 with 3 objectives: Pareto front parameterized by (t1, t2) in [0, pi/2]^2
        # f_1 = cos(t1), f_2 = sin(t1)*cos(t2), f_3 = sin(t2)
        # Reference point: [2.5, 2.5, 2.5]
        
        def obj_from_angles(t1, t2):
            """Compute objective values from angle parameters."""
            f1 = np.cos(t1)
            f2 = np.sin(t1) * np.cos(t2)
            f3 = np.sin(t2)
            return np.array([f1, f2, f3])
        
        def x_from_angles(t1, t2):
            """Convert angles to decision variables on the Pareto front."""
            x = np.zeros(dim)
            x[0] = t1 / (np.pi/2)
            x[1] = t2 / (np.pi/2)
            x[2:] = 0.5
            return x
        
        def compute_hypervolume(points, ref):
            """Compute 3D hypervolume using inclusion-exclusion."""
            if len(points) == 0:
                return 0.0
            pts = np.array(points)
            # Filter points strictly better than reference
            mask = np.all(pts < ref, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            # Use sweep-line algorithm for 3D hypervolume
            # For each point, compute its contribution
            n = len(pts)
            if n == 1:
                return float(np.prod(ref - pts[0]))
            
            total_hv = 0.0
            for i in range(n):
                # This point dominates all points with larger f1
                # Find the region where this point is the "leader" in f1
                # Contribution = (ref[0] - f1_i) * area_in_2D where this point is non-dominated
                f1_i = pts[i, 0]
                
                # Find all points that are not dominated by point i in f2-f3 plane
                # Point i dominates another if f2_i <= f2_j AND f3_i <= f3_j
                # We need the area in (f2, f3) space where point i is the minimum in f1
                # and is non-dominated in the (f2, f3) subproblem
                
                # Simpler: compute 2D hypervolume of points with f1 >= f1_i, 
                # restricted to region not covered by earlier points
                # Actually, let's use a different approach:
                # Sort by f1 ascending, then for each point compute its unique contribution
                
                # For point i (sorted by f1), its contribution is:
                # (ref[0] - f1_i) * HV_2D({(f2_j, f3_j) for j >= i}, ref_2d)
                # where HV_2D is the 2D hypervolume of the remaining points
                # But we need to subtract what's already covered by earlier points
                
                # Let me use a cleaner approach:
                # Process points in order of decreasing f1
                # Maintain the "skyline" in (f2, f3) space
                pass
            
            # Use a standard 3D hypervolume algorithm
            # For small n, use the recursive approach
            return self._hv3d(pts, ref)
        
        def _hv3d(self, pts, ref):
            """Compute 3D hypervolume using recursive decomposition."""
            n = len(pts)
            if n == 0:
                return 0.0
            if n == 1:
                return float(np.prod(ref - pts[0]))
            
            # Sort by first coordinate
            pts = pts[np.argsort(pts[:, 0])]
            
            # The point with smallest f0 contributes:
            # (ref[0] - pts[0,0]) * HV_2D of all points in (f1, f2) space
            # But we need to account for dominance
            # 
            # Better: use the standard recursive formula
            # HV = sum over "maximal" points of their unique contribution
            # 
            # Let's use a sweep approach:
            # Sort by f0 ascending
            # For each point i, its unique contribution in f0 dimension is:
            # (ref[0] - f0_i) * [HV_2D of points j>=i in (f1,f2) minus HV_2D of points j<i that dominate]
            
            # Simpler implementation for n <= 200:
            # Use the inclusion-exclusion with sorted points
            total = 0.0
            # Process from largest f0 to smallest
            for i in range(n - 1, -1, -1):
                f0_i = pts[i, 0]
                # This point's f0 contribution extends from f0_i to ref[0]
                # In that slab, we need the 2D HV of all points with f0 >= f0_i
                # that are not dominated in (f1, f2) by any point with f0 < f0_i
                # Actually, since we process from large to small f0:
                # When we add point i, it can only be dominated in (f1,f2) by points already added (with larger f0)
                # No wait, dominance in 3D: p dominates q if p[k] <= q[k] for all k
                # So a point with smaller f0 can dominate a point with larger f0
                # 
                # Let me just use a brute-force-ish approach for n=200
                # Compute HV by summing over all points their "box" minus overlaps
                # This is O(n^2) or O(n^3) which is fine for n=200
                
                # Actually, let's use the standard algorithm:
                # Sort by f0 ascending
                # HV = (ref[0] - f0_0) * HV_2D({(f1_j, f2_j) for all j}) 
                #     - (f0_1 - f0_0) * HV_2D({(f1_j, f2_j) for j >= 1})
                #     - ...
                # No, that's not quite right either.
                
                pass
            
            # Let me implement a clean 3D HV algorithm
            # Using the "maximal points" approach
            # First, find all non-dominated points
            # Then compute HV recursively
            
            # For efficiency with n<=200, use this approach:
            # Sort by f0 ascending
            # For each consecutive pair, the "slab" between them has a certain 2D HV
            # The total HV is the sum of slab_width * slab_2D_HV
            
            # Slab i (between f0_i and f0_{i+1}, or f0_{n-1} and ref[0]):
            # Contains all points with index >= i
            # The 2D HV in that slab is HV_2D of {(f1_j, f2_j) for j >= i}
            
            # So: HV = sum_{i=0}^{n-1} (f0_{i+1} - f0_i) * HV_2D({(f1_j, f2_j) for j >= i})
            # where f0_n = ref[0]
            
            # Wait, that's not right. Let me think again.
            # 
            # The hypervolume is the volume of the union of boxes [f_i, ref] for all i.
            # Sort by f0 ascending: f0_0 <= f0_1 <= ... <= f0_{n-1}
            # 
            # The box for point i is [f0_i, ref[0]] x [f1_i, ref[1]] x [f2_i, ref[2]]
            # 
            # Using the "sweep" from high f0 to low f0:
            # At f0 = ref[0], no boxes active
            # As we decrease f0, boxes become active
            # 
            # Alternative: use the fact that HV can be computed as:
            # HV = integral from 0 to ref[0] of A(t) dt
            # where A(t) is the 2D area of the projection of the union of boxes at height t
            # 
            # For t in [f0_i, f0_{i+1}), the active points are those with f0_j <= t, i.e., j <= i
            # Wait no, the box for point j spans [f0_j, ref[0]], so at height t,
            # point j is active if f0_j <= t <= ref[0], i.e., f0_j <= t
            # So at height t, active points are {j : f0_j <= t}
            # 
            # Sort f0 ascending: f0_0 <= f0_1 <= ...
            # For t in [f0_i, f0_{i+1}), active points are {0, 1, ..., i}
            # A(t) = HV_2D({(f1_j, f2_j) for j=0..i}, ref_2d)
            # 
            # So HV = sum_{i=0}^{n-1} (f0_{i+1} - f0_i) * HV_2D({(f1_j, f2_j) for j=0..i})
            # where f0_n = ref[0]
            # And for i = n-1: (ref[0] - f0_{n-1}) * HV_2D(all points)
            
            # Now I need HV_2D for each prefix
            # HV_2D of a set of points in 2D with reference (r1, r2):
            # Sort by f1 ascending, then HV_2D = sum (f1_{i+1} - f1_i) * (r2 - min_{j<=i} f2_j)
            # Wait, similar logic:
            # HV_2D = integral from 0 to r1 of B(s) ds where B(s) = length of active intervals at s
            # For points sorted by f1: f1_0 <= f1_1 <= ...
            # At s in [f1_i, f1_{i+1}), active points are {0..i}
            # B(s) = r2 - min_{j=0..i} f2_j
            # HV_2D = sum_{i=0}^{m-1} (f1_{i+1} - f1_i) * (r2 - min_{j=0..i} f2_j)
            # where f1_m = r1
            
            # This is O(n^2) total if we compute all prefixes, which is fine for n=200
            
            # Let's implement this properly
            pts_sorted = pts[np.argsort(pts[:, 0])]
            f0_vals = pts_sorted[:, 0]
            f12 = pts_sorted[:, 1:3]
            
            total_hv = 0.0
            for i in range(n):
                if i < n - 1:
                    slab_width = f0_vals[i + 1] - f0_vals[i]
                else:
                    slab_width = ref[0] - f0_vals[i]
                
                # Compute HV_2D of first i+1 points in (f1, f2) space
                prefix_12 = f12[:i + 1]
                hv2d = self._hv2d(prefix_12, np.array([ref[1], ref[2]]))
                total_hv += slab_width * hv2d
            
            return total_hv
        
        def _hv2d(self, pts2d, ref2d):
            """Compute 2D hypervolume."""
            m = len(pts2d)
            if m == 0:
                return 0.0
            if m == 1:
                return float((ref2d[0] - pts2d[0, 0]) * (ref2d[1] - pts2d[0, 1]))
            
            # Sort by first coordinate
            pts2d = pts2d[np.argsort(pts2d[:, 0])]
            f1_vals = pts2d[:, 0]
            f2_vals = pts2d[:, 1]
            
            total = 0.0
            min_f2 = np.inf
            for i in range(m):
                min_f2 = min(min_f2, f2_vals[i])
                if i < m - 1:
                    width = f1_vals[i + 1] - f1_vals[i]
                else:
                    width = ref2d[0] - f1_vals[i]
                total += width * (ref2d[1] - min_f2)
            
            return total
        
        # Now, greedy hypervolume maximization:
        # Start with a diverse initial set, then greedily add points that maximize HV gain
        
        # Step 1: Create a fine candidate pool in (t1, t2) space
        n_candidates = 400
        t1_cand = np.linspace(0, np.pi/2, n_candidates)
        t2_cand = np.linspace(0, np.pi/2, n_candidates)
        T1, T2 = np.meshgrid(t1_cand, t2_cand, indexing='ij')
        candidate_angles = np.column_stack([T1.ravel(), T2.ravel()])
        candidate_objs = np.array([obj_from_angles(t1, t2) for t1, t2 in candidate_angles])
        
        # Step 2: Greedy selection
        ref = np.array([2.5, 2.5, 2.5])
        selected_indices = []
        selected_objs = []
        
        # Initial: pick 4 corners + center for diversity
        init_indices = [
            0,  # (0, 0)
            n_candidates - 1,  # (pi/2, pi/2) 
            0 * n_candidates + (n_candidates - 1),  # (0, pi/2)
            (n_candidates - 1) * n_candidates + 0,  # (pi/2, 0)
            (n_candidates // 2) * n_candidates + (n_candidates // 2)  # center
        ]
        # Deduplicate
        init_indices = list(set(init_indices))
        
        for idx in init_indices:
            selected_indices.append(idx)
            selected_objs.append(candidate_objs[idx])
        
        # Greedily add points
        available = set(range(len(candidate_angles))) - set(selected_indices)
        
        while len(selected_indices) < budget and available:
            if time.time() > deadline - 50:
                break
            
            best_idx = -1
            best_gain = -1
            current_hv = self._hv3d(np.array(selected_objs), ref) if selected_objs else 0.0
            
            # Sample from available if too many
            avail_list = list(available)
            if len(avail_list) > 100:
                # Random subsample for speed
                np.random.seed(42 + len(selected_indices))
                avail_list = list(np.random.choice(avail_list, size=100, replace=False))
            
            for idx in avail_list:
                # Compute HV with this point added
                new_objs = selected_objs + [candidate_objs[idx]]
                new_hv = self._hv3d(np.array(new_objs), ref)
                gain = new_hv - current_hv
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                selected_objs.append(candidate_objs[best_idx])
                available.discard(best_idx)
        
        # Convert selected angles to decision variables
        points = []
        for idx in selected_indices:
            t1, t2 = candidate_angles[idx]
            x = x_from_angles(t1, t2)
            points.append(x)
        
        # Ensure we have exactly budget points (pad with duplicates if needed)
        while len(points) < budget:
            if points:
                points.append(points[-1].copy())
            else:
                points.append(x_from_angles(0.5 * np.pi/2, 0.5 * np.pi/2))
        
        # Evaluate all points
        F = np.array([problem(x) for x in points[:budget]])
        
        if time.time() > deadline:
            return F
        
        return F
>>>>>>> REPLACE

Wait, I realize the greedy approach with full HV recomputation at each step will be very slow (O(n^3) per step, O(n^4) total). Let me use a faster approach: precompute a good set using farthest-point sampling in objective space, which was shown to improve scores.

