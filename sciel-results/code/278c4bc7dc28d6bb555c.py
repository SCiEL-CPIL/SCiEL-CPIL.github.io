"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hypervolume-driven adaptive multi-start local search for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, F, ref):
        """Compute hypervolume indicator for 3 objectives using exact 3D algorithm."""
        # Filter points strictly better than reference
        mask = np.all(F < ref, axis=1)
        F = F[mask]
        if len(F) == 0:
            return 0.0
        
        # Sort by first objective
        order = np.argsort(F[:, 0])
        F = F[order]
        
        # 3D hypervolume: for each point, compute contribution
        # Contribution of point i is (ref[0]-F[i][0]) * HV2D(F[i][1:], ref[1:]) 
        # where HV2D considers only points j > i (in sorted order) that dominate in remaining dims
        # Actually, standard 3D HV:
        # Sort by f1. For each point i, the volume it uniquely contributes is:
        # (ref[0] - f1[i]) * (max(0, min(ref[1], f1_next_bound) - f1[i])) ... this is complex.
        
        # Simpler: use the standard O(n^2) or O(n log n) 3D HV algorithm.
        # For 3 objectives, we can sort by f1, then for each point, compute the 2D HV of the remaining points
        # projected onto (f2, f3) that are non-dominated by the current point in f2, f3.
        
        n = len(F)
        total_hv = 0.0
        # Precompute for efficiency? n is small (<=200), so O(n^2) is fine.
        
        # Sort by f1 ascending
        F_sorted = F  # already sorted
        
        for i in range(n):
            f1 = F_sorted[i, 0]
            # Points j > i have f1_j >= f1_i
            # The volume contributed by point i is the box from (f1_i, f2_i, f3_i) to (ref[0], max_f2, max_f3)
            # but clipped by other points.
            # Actually, the standard formula:
            # HV = sum over i of (ref[0] - f1[i]) * HV2D({(f2[j], f3[j]) for j>i and f2[j] < f2[i] or f3[j] < f3[i]} union {(ref[1], ref[3])})
            # This is getting complicated. Let's use a simpler incremental approach or a library-like exact 3D HV.
            
            # Exact 3D HV algorithm (from literature):
            # Sort points by f1.
            # For each point i, let S_i be the set of points j > i that are non-dominated by i in (f2, f3).
            # The contribution is (ref[0] - f1[i]) * HV2D(S_i, ref[1:])
            # But S_i is not just all j>i; it's the Pareto front of {F[j] for j>i} intersected with the region dominated by i in f2,f3?
            # No. The standard way:
            # HV = sum_{i} (ref[0] - f1[i]) * HV2D(P_i, ref[1:]), where P_i is the set of points j>i that are non-dominated by any point k in [i, j-1] in the (f2,f3) space? 
            # Actually, the simplest correct way for small n:
            # For each point i, compute the volume of the box [f1[i], ref[0]] x [f2[i], ref[1]] x [f3[i], ref[2]] that is NOT covered by any other point.
            # This is equivalent to: 
            # Volume = (ref[0]-f1[i]) * (ref[1]-f2[i]) * (ref[2]-f3[i]) - overlaps.
            
            # Let's use the standard O(n^2) 3D HV algorithm:
            # Sort by f1.
            # For i from 0 to n-1:
            #   Let f1_i = F[i, 0]
            #   The region dominated by i is [f1_i, ref[0]] x [f2_i, ref[1]] x [f3_i, ref[2]]
            #   But we must subtract regions already dominated by points j < i (which have smaller f1).
            #   Actually, it's easier to iterate in reverse or use the "slice" method.
            
            pass  # Placeholder, will implement below

        # Let's implement a known correct 3D HV algorithm.
        # Reference: "A Fast Algorithm for Computing Hypervolume" 
        # For 3 objectives:
        # 1. Sort points by f1 ascending.
        # 2. For each point i, compute the 2D hypervolume of the set of points j > i that are non-dominated by i in (f2, f3), 
        #    but actually, the contribution of point i is (ref[0] - f1[i]) * HV2D({ (f2[j], f3[j]) for j > i } filtered to those non-dominated in 2D by i? No.
        
        # Correct algorithm:
        # HV = sum_{i=0}^{n-1} (ref[0] - f1[i]) * HV2D(P_i, ref[1:])
        # where P_i is the Pareto front of { F[j] for j > i } in the (f2, f3) subspace, 
        # but restricted to points that are not dominated by F[i] in (f2, f3)? 
        # No, the standard formula is:
        # The volume dominated by point i but not by any point j < i is:
        # (ref[0] - f1[i]) * HV2D({ (f2[j], f3[j]) for j > i and (f2[j] > f2[i] or f3[j] > f3[i]) }, ref[1:])
        # This is not quite right either.
        
        # Let's use the simplest correct O(n^3) or O(n^2) method for n<=200.
        # For each point i, compute the volume of the box B_i = [f1[i], ref[0]] x [f2[i], ref[1]] x [f3[i], ref[2]]
        # that is not covered by any other point j.
        # A point j covers part of B_i if f1[j] <= f1[i] (impossible since sorted, j>i has f1[j]>=f1[i]) 
        # Actually, if sorted by f1 ascending, point j > i has f1[j] >= f1[i]. So j cannot dominate i in f1.
        # Therefore, the only overlaps are from points j < i (which have f1[j] <= f1[i]).
        # So, the unique contribution of point i is the volume of B_i not covered by points j < i.
        
        # This is still complex. Let's use a different approach:
        # Since n is small (<=200), we can use the "inclusion-exclusion" or simply compute the union of boxes.
        # But 3D union of boxes is hard.
        
        # Alternative: Use the fact that the Pareto front in 3D can be computed, and then HV is the sum of volumes of the "cells" formed by the front.
        # For the purpose of this optimizer, we only need a relative measure to select the top 10 points.
        # We can use a simpler approximation: for each point, compute its "contribution" as the product of its distances to the reference point, 
        # weighted by how non-dominated it is. But this is not accurate.
        
        # Let's implement a correct 3D HV using the standard algorithm from "Hypervolume Indicator" literature.
        # Algorithm:
        # 1. Sort points by f1 ascending.
        # 2. Initialize a list of "active" points for 2D HV.
        # 3. Iterate i from n-1 down to 0:
        #    - The contribution of point i is (ref[0] - f1[i]) * HV2D(active_points, ref[1:])
        #    - Then, add point i to active_points, but remove any points dominated by i in (f2, f3).
        #    - HV2D is computed for the 2D Pareto front of active_points.
        
        # This is O(n^2) or O(n^3) but acceptable for n=200.
        
        # Implement HV2D for a set of 2D points:
        def hv2d(points_2d, ref2):
            # points_2d: list of (f2, f3)
            # ref2: (ref[1], ref[2])
            if len(points_2d) == 0:
                return 0.0
            # Sort by f2 ascending
            points_2d = sorted(points_2d, key=lambda p: p[0])
            hv = 0.0
            last_f3 = ref2[1]
            for i in range(len(points_2d) - 1, -1, -1):
                f2, f3 = points_2d[i]
                if f3 < last_f3:
                    hv += (ref2[0] - f2) * (last_f3 - f3)
                    last_f3 = f3
                else:
                    # This point is dominated by a previous point in the sorted order? 
                    # Actually, if sorted by f2 ascending, and we iterate backwards, 
                    # we need to keep track of the minimum f3 seen so far (which is the "frontier").
                    # If f3 >= last_f3, it is dominated by the point that set last_f3.
                    pass
            return hv
        
        # Correct HV2D:
        def hv2d_correct(points_2d, ref2):
            if len(points_2d) == 0:
                return 0.0
            # Sort by f2 ascending
            points_2d = sorted(points_2d, key=lambda p: p[0])
            # Remove dominated points: keep only the Pareto front
            front = []
            min_f3 = ref2[1]
            for p in reversed(points_2d):
                if p[1] < min_f3:
                    front.append(p)
                    min_f3 = p[1]
            # Compute HV for the front
            hv = 0.0
            for i in range(len(front)):
                f2, f3 = front[i]
                if i == len(front) - 1:
                    next_f2 = ref2[0]
                else:
                    next_f2 = front[i+1][0]
                hv += (next_f2 - f2) * (ref2[1] - f3)
            return hv

        # Now, 3D HV:
        F_sorted = F  # sorted by f1
        n = len(F_sorted)
        active_points = []  # list of (f2, f3)
        total_hv = 0.0
        
        for i in range(n - 1, -1, -1):
            f1, f2, f3 = F_sorted[i]
            # Compute HV2D of active_points
            hv2 = hv2d_correct(active_points, (ref[1], ref[2]))
            total_hv += (ref[0] - f1) * hv2
            
            # Add (f2, f3) to active_points, removing dominated points
            # A point (a2, a3) in active_points is dominated by (f2, f3) if f2 <= a2 and f3 <= a3
            new_active = []
            for p in active_points:
                if not (f2 <= p[0] and f3 <= p[1]):
                    new_active.append(p)
            new_active.append((f2, f3))
            active_points = new_active
            
        return total_hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3.0, 5.0, 7.0])
        
        n_evals = 0
        budget = self.budget
        
        # Initialize 50 diverse points
        n_init = 50
        X = np.random.uniform(lo, hi, size=(n_init, self.dim))
        F = np.array([problem(x) for x in X])
        n_evals += n_init
        
        # Store all evaluated points
        all_X = X.copy()
        all_F = F.copy()
        
        # Iteratively polish top 10 points by hypervolume contribution
        # We will use the remaining budget for local search steps
        remaining_budget = budget - n_evals
        
        # We'll do multiple rounds of polishing
        # Each round: select top 10 by HV contribution, generate neighbors, evaluate, keep best
        n_rounds = min(remaining_budget // 10, 20)  # At most 20 rounds, each using ~10 evals
        
        for r in range(n_rounds):
            if time.time() > deadline:
                break
            if n_evals >= budget:
                break
            
            # Compute HV contribution for each point
            # Approximate: for each point, compute the HV of the set of points it dominates? 
            # Simpler: compute the HV of the entire set, then for each point, compute the HV without it, and take the difference.
            # This is O(n^2 * HV_cost) which is too slow for n=200.
            # Alternative: use a greedy selection: select points that are non-dominated and have high "spread".
            # For simplicity, let's select the top 10 non-dominated points with the best "distance" to the ideal point.
            
            # Find non-dominated points
            mask_nd = np.ones(len(all_F), dtype=bool)
            for i in range(len(all_F)):
                for j in range(len(all_F)):
                    if i != j:
                        if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                            mask_nd[i] = False
                            break
            
            nd_indices = np.where(mask_nd)[0]
            if len(nd_indices) == 0:
                break
            
            # Select top 10 from non-dominated points based on hypervolume contribution
            # Approximate contribution: product of (ref - F[i])
            contributions = np.prod(ref - all_F[nd_indices], axis=1)
            top_k = min(10, len(nd_indices))
            top_indices = nd_indices[np.argsort(contributions)[-top_k:]]
            
            # Generate neighbors for each top point
            step_size = 0.1 * (hi - lo)
            n_neighbors = max(1, (budget - n_evals) // len(top_indices))
            
            for idx in top_indices:
                if n_evals >= budget:
                    break
                x = all_X[idx]
                # Generate one neighbor
                neighbor = x + np.random.uniform(-step_size, step_size, size=self.dim)
                neighbor = np.clip(neighbor, lo, hi)
                
                f_neighbor = problem(neighbor)
                n_evals += 1
                
                # If neighbor is non-dominated, add to population
                # Check if neighbor is dominated by any existing point
                dominated = False
                for i in range(len(all_F)):
                    if np.all(all_F[i] <= f_neighbor) and np.any(all_F[i] < f_neighbor):
                        dominated = True
                        break
                
                if not dominated:
                    all_X = np.vstack([all_X, [neighbor]])
                    all_F = np.vstack([all_F, [f_neighbor]])
                    # Remove points dominated by neighbor
                    new_mask = np.ones(len(all_F), dtype=bool)
                    for i in range(len(all_F)):
                        if np.all(f_neighbor <= all_F[i]) and np.any(f_neighbor < all_F[i]):
                            new_mask[i] = False
                    all_X = all_X[new_mask]
                    all_F = all_F[new_mask]
        
        return all_X
# EVOLVE-BLOCK-END
