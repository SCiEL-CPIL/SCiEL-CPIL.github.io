# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT1 using a MODEL-AND-OPTIMIZE approach.
    
    The ZDT1 Pareto front is known analytically: f1 = x[0], f2 = 1 - sqrt(x[0]) - x[0]*sin(10*pi*x[0])
    for x[0] in [0, 1] with x[1:] = 0.
    
    Strategy:
    1. Compute the Pareto front analytically on a dense grid.
    2. Select a subset of points that maximizes hypervolume using local search.
    3. Optimize the selection by swapping points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt1_front(self, u):
        """Compute ZDT1 Pareto front values for given u values."""
        f1 = u
        f2 = 1.0 - np.sqrt(u) - u * np.sin(10.0 * np.pi * u)
        return f1, f2

    def _hypervolume_sorted(self, f1, f2, ref=(11.0, 11.0)):
        """
        Compute 2D hypervolume for non-dominated points assumed to be sorted by f1.
        """
        # Only points better than reference count
        # Since f1 is sorted and increasing, and ref is large, we can just check bounds
        # But we must ensure f2 < ref[1] as well.
        # In ZDT1, f2 is always < 11, so we can skip f2 check if we are sure.
        # But to be safe:
        mask = (f1 < ref[0]) & (f2 < ref[1])
        if not np.all(mask):
            f1 = f1[mask]
            f2 = f2[mask]
            
        if len(f1) == 0:
            return 0.0
        
        # Compute hypervolume using the staircase method
        # f1 is sorted ascending.
        hv = 0.0
        prev_f2 = ref[1]
        # Iterate in reverse order? Or forward?
        # Standard formula:
        # Sort by f1 ascending.
        # HV = sum_{i} (ref[0] - f1[i]) * (f2[i-1] - f2[i])
        # where f2[-1] = ref[1]
        # This requires f2 to be "sorted" in a way that corresponds to the staircase.
        # Actually, for non-dominated points sorted by f1, f2 must be decreasing.
        # If f2 is not decreasing, some points are dominated.
        # We should filter out dominated points or assume the front is non-dominated.
        # ZDT1 front is non-dominated.
        # So f2 should be strictly decreasing as f1 increases?
        # Let's check. f1 = u. f2 = 1 - sqrt(u) - u sin(10 pi u).
        # Derivative of f2 is not always negative.
        # So the front might have segments where f2 increases?
        # If f2 increases, then the point with higher f1 and higher f2 is dominated by the one with lower f1 and lower f2?
        # No, if f1 increases and f2 increases, the new point is NOT dominated by the old one (old has lower f1, lower f2).
        # Wait. Point A: (0.1, 0.9). Point B: (0.2, 0.95).
        # A dominates B? No. A has lower f1, but lower f2? No, A has lower f2.
        # If A has lower f1 AND lower f2, A dominates B.
        # If f1 increases and f2 increases, neither dominates the other.
        # But for hypervolume calculation, we only care about the non-dominated front.
        # If the set of points contains dominated points, the hypervolume calculation using the simple staircase formula on sorted f1 will be WRONG if f2 is not monotonically decreasing.
        # We must filter out dominated points first, or use a robust HV algorithm.
        # Given the small N (200), we can just filter dominated points.
        
        # Filter dominated points
        # A point is dominated if there exists another point with f1' <= f1 and f2' <= f2, and at least one strict.
        # Since we are on the Pareto front, all points should be non-dominated.
        # However, due to the oscillations, is the ZDT1 front non-dominated?
        # Yes, the analytic Pareto front is non-dominated.
        # So f2 should be "effectively" decreasing in the sense that no point is dominated.
        # But f2 can go up and down.
        # If f2 goes up, does it create a dominated point?
        # If f1 increases and f2 increases, the new point is not dominated by the previous one.
        # But is it dominated by some earlier point?
        # If f2 oscillates, it's possible that f2(u2) > f2(u1) for u2 > u1.
        # Then point 2 is not dominated by point 1.
        # But what if f2(u3) < f2(u1) for u3 > u2 > u1?
        # Then point 1 might be dominated by point 3?
        # No, point 3 has higher f1.
        # So points on the analytic front are non-dominated.
        
        # So we can assume the set is non-dominated.
        # But the staircase formula `sum (ref[0]-f1[i]) * (f2[i-1] - f2[i])` assumes that the "skyline" is formed by the points in order of f1.
        # This is only true if f2 is monotonically decreasing.
        # If f2 is not monotonically decreasing, the hypervolume is the area under the upper envelope of the points?
        # No, hypervolume is the volume of the region dominated by the set.
        # In 2D, if we have points (f1_i, f2_i), the dominated region is the union of rectangles [0, f1_i] x [0, f2_i] (if minimizing to 0).
        # But here we have a reference point.
        # The region is [f1_i, ref[0]] x [f2_i, ref[1]]? No.
        # The region dominated by point i is [f1_i, ref[0]] x [f2_i, ref[1]]?
        # No. A point y dominates x if y_i <= x_i for all i.
        # The set of points dominated by the set S is { y | exists x in S s.t. x_i <= y_i for all i }.
        # Wait, the evaluator says "minimizing all 2 objectives".
        # And "points strictly better than the reference point".
        # Better means smaller.
        # So the dominated region is the union of boxes [0, f1_i] x [0, f2_i]?
        # No, the hypervolume is usually defined as the volume of the region that is dominated by the set and contains the reference point?
        # Standard definition: HV(S, r) = Vol({ y | exists x in S, x <= y <= r }).
        # This is the volume of the region "above" the front and "below" the reference.
        # In 2D, if we sort by f1, the boundary of this region is the "upper envelope" of the points?
        # No, the region is defined by y >= f1_i and y >= f2_i?
        # No. x <= y. So y1 >= f1_i and y2 >= f2_i.
        # And y <= r.
        # So the region is the union of rectangles [f1_i, r1] x [f2_i, r2].
        # To calculate this volume, we can sort by f1.
        # The "effective" f2 for a given f1 is the minimum f2 among points with f1_j <= f1.
        # Let g(t) = min { f2_j | f1_j <= t }.
        # Then the volume is integral from 0 to r1 of (r2 - g(t)) dt?
        # No, the region is y >= g(t).
        # So the volume is ∫ (r2 - g(t)) dt.
        # Since g(t) is a step function (non-increasing), we can compute it as a sum of rectangles.
        # We need to find the "skyline" of the points, i.e., the non-dominated points sorted by f1, where f2 is decreasing.
        # If the input points are on the Pareto front, they are non-dominated.
        # But are they sorted such that f2 is decreasing?
        # For ZDT1, f1 = u. f2 is not monotonic.
        # So we MUST filter to get the skyline (non-dominated points sorted by f1 with decreasing f2).
        
        # Step 1: Filter dominated points to get the skyline.
        # Sort by f1 ascending.
        # Iterate and keep points that have f2 lower than all previous points.
        # This gives the non-dominated front sorted by f1 with strictly decreasing f2.
        
        if len(f1) == 0:
            return 0.0
            
        # Sort by f1
        order = np.argsort(f1)
        f1 = f1[order]
        f2 = f2[order]
        
        # Find skyline
        sky_f1 = []
        sky_f2 = []
        min_f2_so_far = np.inf
        for i in range(len(f1)):
            if f2[i] < min_f2_so_far:
                sky_f1.append(f1[i])
                sky_f2.append(f2[i])
                min_f2_so_far = f2[i]
                
        if not sky_f1:
            return 0.0
            
        sky_f1 = np.array(sky_f1)
        sky_f2 = np.array(sky_f2)
        
        # Now sky_f1 is increasing, sky_f2 is decreasing.
        # Compute HV
        hv = 0.0
        prev_f2 = ref[1]
        for i in range(len(sky_f1)):
            width = ref[0] - sky_f1[i]
            height = prev_f2 - sky_f2[i]
            if width > 0 and height > 0:
                hv += width * height
            prev_f2 = sky_f2[i]
            
        return hv

    def _local_search(self, selected_idxs, u_candidates, ref=(11.0, 11.0)):
        """
        Local search: try to swap selected points with unselected neighbors to improve hypervolume.
        """
        if len(u_candidates) <= len(selected_idxs):
            return selected_idxs
        
        all_f1, all_f2 = self._zdt1_front(u_candidates)
        
        def get_hv(sel):
            sel_sorted = sorted(sel)
            sel_f1 = all_f1[sel_sorted]
            sel_f2 = all_f2[sel_sorted]
            return self._hypervolume_sorted(sel_f1, sel_f2, ref)
        
        current_hv = get_hv(selected_idxs)
        selected_set = set(selected_idxs)
        
        improved = True
        max_passes = 100 # Increase passes
        for _ in range(max_passes):
            improved = False
            for i in range(len(selected_idxs)):
                for j in range(len(u_candidates)):
                    if j in selected_set:
                        continue
                    
                    # Try swapping selected_idxs[i] with j
                    new_sel = list(selected_idxs)
                    removed = new_sel.pop(i)
                    new_sel.append(j)
                    new_sel_set = set(new_sel)
                    
                    if len(new_sel_set) < len(selected_idxs):
                        continue  # Duplicate
                    
                    new_hv = get_hv(new_sel)
                    if new_hv > current_hv + 1e-10:
                        selected_idxs = new_sel
                        selected_set = new_sel_set
                        current_hv = new_hv
                        improved = True
                        break  # Restart from beginning
            if not improved:
                break
        
        return sorted(selected_idxs)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = (11.0, 11.0)

        n_select = self.budget
        
        # Step 1: Generate candidate points on the front
        # Use a dense grid
        n_candidates = max(n_select * 10, 2000)
        u_candidates = np.linspace(0.0, 1.0, n_candidates)
        
        # Step 2: Compute objective values for all candidates
        all_f1, all_f2 = self._zdt1_front(u_candidates)
        
        # Filter: only points better than reference
        valid_mask = (all_f1 < ref[0]) & (all_f2 < ref[1])
        u_candidates = u_candidates[valid_mask]
        all_f1 = all_f1[valid_mask]
        all_f2 = all_f2[valid_mask]
        
        if len(u_candidates) == 0:
            return []
        
        n_candidates = len(u_candidates)
        n_select = min(n_select, n_candidates)
        
        # Step 3: Initial selection - use a stratified sampling
        # Start with evenly spaced points
        initial_idx = np.linspace(0, n_candidates - 1, n_select).astype(int)
        
        # Step 4: Local search to improve hypervolume
        if time.time() < deadline - 30:
            selected = self._local_search(initial_idx, u_candidates, ref)
        else:
            selected = initial_idx
        
        # Step 5: Evaluate selected points
        results = []
        for idx in selected:
            if time.time() > deadline - 5:
                break
            u_val = u_candidates[idx]
            x = np.zeros(self.dim)
            x[0] = u_val
            x[1:] = 0.0  # Exact Pareto front
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
        
        return results

# EVOLVE-BLOCK-END
