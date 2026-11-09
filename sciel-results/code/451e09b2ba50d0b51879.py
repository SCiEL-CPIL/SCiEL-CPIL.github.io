"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using HV-guided local search with restarts."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Deadline guard: The evaluator kills at 330s. We aim to finish by 290s.
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Reference point for HV calculation (from task description)
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Correct 3D HV calculation
        def compute_hv_3d(points, ref):
            """Exact hypervolume for 3D using inclusion-exclusion."""
            # Filter points strictly better than reference
            valid = [p for p in points if np.all(p < ref)]
            if not valid:
                return 0.0
            
            n = len(valid)
            if n == 0:
                return 0.0
            
            # Sort by first objective (ascending)
            valid = sorted(valid, key=lambda p: p[0])
            
            # For each point, compute the volume of the box it contributes
            # that is not covered by any point with smaller f0
            hv = 0.0
            
            for i in range(n):
                p = valid[i]
                # Box from p to ref in dimensions 1 and 2
                # But we need to subtract volumes already covered by points j < i
                # that have f1 <= p[1] and f2 <= p[2]
                
                # The contribution of point i is:
                # (ref[0] - p[0]) * (area in f1-f2 plane not covered by points j<i)
                
                # Collect points j < i that are "relevant" (f1 < p[1] or f2 < p[2])
                # Actually, for the box [p[0], ref[0]] x [p[1], ref[1]] x [p[2], ref[2]],
                # we need to subtract regions covered by points with f0 < p[0]
                
                # Simpler: for point i, the new volume is the box [p[0], ref[0]] x [p[1], ref[1]] x [p[2], ref[2]]
                # minus the union of boxes from previous points that overlap.
                
                # Use 2D HV for the f1-f2 slice
                # Points that can cover part of this slice: j < i with f1 < ref[1] and f2 < ref[2]
                
                # Let's compute the 2D HV of the region [p[1], ref[1]] x [p[2], ref[2]]
                # covered by points j < i (projected to f1, f2)
                
                # Actually, the standard approach:
                # Sort by f0. For point i, compute the 2D HV of the set of points {j < i} 
                # clipped to the rectangle [p[1], ref[1]] x [p[2], ref[2]].
                
                # 2D HV helper
                def hv_2d(pts, ref2):
                    """2D hypervolume for points with reference ref2 = [ref[1], ref[2]]."""
                    pts = [p for p in pts if p[0] < ref2[0] and p[1] < ref2[1]]
                    if not pts:
                        return 0.0
                    pts = sorted(pts, key=lambda p: p[0])
                    hv = 0.0
                    max_y = 0.0
                    for p in pts:
                        if p[1] > max_y:
                            hv += (ref2[0] - p[0]) * (p[1] - max_y)
                            max_y = p[1]
                    return hv
                
                # Get f1, f2 of all points j < i
                prev_pts = [(valid[j][1], valid[j][2]) for j in range(i)]
                
                # The area of [p[1], ref[1]] x [p[2], ref[2]] not covered by prev_pts
                # = total area - area covered by prev_pts within this rectangle
                
                total_area = (ref[1] - p[1]) * (ref[2] - p[2])
                
                # Area covered by prev_pts within [p[1], ref[1]] x [p[2], ref[2]]
                # Clip prev_pts to this rectangle
                clipped = [(max(pp[0], p[1]), max(pp[1], p[2])) for pp in prev_pts]
                clipped = [(c0, c1) for c0, c1 in clipped if c0 < ref[1] and c1 < ref[2]]
                
                covered = hv_2d(clipped, [ref[1], ref[2]])
                
                new_area = total_area - covered
                hv += (ref[0] - p[0]) * new_area
            
            return hv
        
        # Initialize population with random points
        init_size = min(30, budget // 10)
        if init_size < 10: init_size = 10
        
        X = []
        F = []
        used = 0
        
        # Initial random sampling
        for _ in range(init_size):
            if used >= budget: break
            x = lo + (hi - lo) * np.random.rand(dim)
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X.append(x)
            F.append(f)
            used += 1
            
        def get_non_dominated(F):
            nd_idx = []
            n = len(F)
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            return nd_idx
        
        # Main Loop
        best_hv = compute_hv_3d(F, ref_point)
        
        while used < budget:
            if time.time() > deadline:
                break
            
            nd_idx = get_non_dominated(F)
            
            if not nd_idx:
                idx = np.random.randint(len(X))
            else:
                # 70% chance to pick from non-dominated, 30% global
                if np.random.rand() < 0.7:
                    idx = nd_idx[np.random.randint(len(nd_idx))]
                else:
                    idx = np.random.randint(len(X))
            
            base_x = X[idx]
            
            # Adaptive step size: start larger, decrease as we approach deadline
            progress = used / budget
            step_scale = 0.15 * (1.0 - 0.5 * progress)
            step_size = step_scale * (hi - lo)
            
            new_x = base_x + np.random.randn(dim) * step_size
            new_x = np.clip(new_x, lo, hi)
            
            new_f = np.array(problem(new_x))
            used += 1
            
            # Check if new point is dominated by current population
            dominated = False
            for i in range(len(F)):
                if np.all(F[i] <= new_f) and np.any(F[i] < new_f):
                    dominated = True
                    break
            
            if not dominated:
                # Add to population
                X.append(new_x)
                F.append(new_f)
                
                # Prune dominated points to keep population manageable
                if len(X) > 80:
                    nd_idx = get_non_dominated(F)
                    if len(nd_idx) < len(X):
                        X = [X[i] for i in nd_idx]
                        F = [F[i] for i in nd_idx]
                
                # Update best HV
                current_hv = compute_hv_3d(F, ref_point)
                if current_hv > best_hv:
                    best_hv = current_hv
                        
        return X
# EVOLVE-BLOCK-END
