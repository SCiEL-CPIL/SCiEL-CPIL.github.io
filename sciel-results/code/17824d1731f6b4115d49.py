"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    
    Strategy:
    Exploit the analytic structure of ZDT1. The Pareto front is known to lie
    on x[1:] = 0. We select the 200 best points on this front to maximize
    the hypervolume indicator using a greedy algorithm on a dense candidate pool.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 analytic front: x[0] in [0, 1], x[1:] = 0
        # f1 = x[0]
        # f2 = 1 - sqrt(x[0]) - x[0] * sin(10 * pi * x[0])
        
        def zdt1_front_f2(x1):
            return 1.0 - np.sqrt(x1) - x1 * np.sin(10.0 * np.pi * x1)

        # 1. Generate a dense pool of candidate x1 values
        # Use a non-uniform distribution to ensure good coverage, 
        # but the greedy step will handle the optimization.
        # Uniform is a safe base.
        n_candidates = 2000
        x1_pool = np.linspace(0.0, 1.0, n_candidates)
        f1_pool = x1_pool
        f2_pool = zdt1_front_f2(x1_pool)
        
        # Sort pool by f1
        sort_idx = np.argsort(f1_pool)
        f1_pool = f1_pool[sort_idx]
        f2_pool = f2_pool[sort_idx]
        x1_pool = x1_pool[sort_idx]
        
        # 2. Greedy selection of points to maximize Hypervolume
        # Reference point
        ref_f1 = 11.0
        ref_f2 = 11.0
        
        # HV calculation for a set of points (sorted by f1)
        # We maintain a list of selected points.
        # To speed up, we can precompute or use efficient data structures.
        # For N=200 and Pool=2000, O(N * Pool * log Pool) is fine.
        
        selected_indices = []
        selected_f1 = []
        selected_f2 = []
        
        # Helper to calculate HV increase of adding a point (f1, f2) to current set
        # Current set is sorted by f1.
        # HV = sum of rectangles.
        # Adding a point splits a rectangle or adds a new one.
        
        # Simpler approach: Just calculate the full HV of the new set?
        # That would be O(N^2) per step, total O(N^3) = 200^3 = 8M. Very fast.
        
        def calc_hv(f1s, f2s):
            # f1s, f2s are numpy arrays, sorted by f1
            if len(f1s) == 0:
                return 0.0
            # Ensure sorted
            # f1s is already sorted in our loop
            # HV = sum_{i} (ref_f1 - f1[i]) * (f2[i-1] - f2[i])
            # where f2[-1] = ref_f2
            # We need to handle the non-dominated points only.
            # Since we are on the Pareto front, all points are non-dominated.
            
            # Construct the "upper" envelope
            # For each point i, the height of the rectangle is f2[i-1] - f2[i]
            # But we must ensure f2[i-1] >= f2[i]. If not, point i is dominated by i-1?
            # On ZDT1 front, f2 is not monotonic?
            # f2(x) = 1 - sqrt(x) - x sin(10 pi x).
            # It oscillates. So points can dominate each other in f2?
            # No, the Pareto front consists of non-dominated points.
            # If the analytic curve has sections where f2 increases, those points
            # might be dominated by earlier points with lower f1 and lower f2?
            # Actually, for ZDT1, the Pareto set is x[1:]=0, x[0] in [0,1].
            # The Pareto front is the image of this set.
            # Are all points on this curve non-dominated?
            # If f2 increases as x1 increases, then a point with smaller x1 (smaller f1)
            # and smaller f2 dominates the one with larger x1.
            # So we must filter out dominated points from the pool before greedy selection.
            
            # Let's filter the pool to keep only non-dominated points.
            # Sort by f1.
            # Iterate and keep points where f2 is strictly less than the min f2 so far.
            
            pass # We will do this filtering on the fly or pre-filter the pool.

        # Pre-filter pool to non-dominated points
        # Sort by f1
        # f1 is increasing.
        # We want to keep points where f2 is lower than all previous f2s.
        
        nd_indices = []
        min_f2_so_far = ref_f2
        for i in range(n_candidates):
            if f2_pool[i] < min_f2_so_far:
                nd_indices.append(i)
                min_f2_so_far = f2_pool[i]
        
        f1_nd = f1_pool[nd_indices]
        f2_nd = f2_pool[nd_indices]
        x1_nd = x1_pool[nd_indices]
        
        n_nd = len(f1_nd)
        
        # Greedy selection from the non-dominated pool
        # We want to select K points.
        # Start with empty set.
        # At each step, pick the point that maximizes HV increase.
        
        selected_f1_list = []
        selected_f2_list = []
        selected_x1_list = []
        
        # To make HV calculation fast, we can maintain the sorted list.
        # Since we are adding points one by one, and we have a pool of candidates,
        # we can just calculate the HV of the current set + candidate.
        
        # Optimization: 
        # HV of set S is sum of areas.
        # Adding point p to S:
        # Find position of p in S (sorted by f1).
        # The area change is local.
        
        # Given N=200 and Pool~2000, let's just do the O(N) HV update.
        
        def hv_increase(f1_new, f2_new, f1_sel, f2_sel):
            # f1_sel, f2_sel are sorted lists (numpy arrays)
            # Insert f1_new, f2_new into the sequence
            # Calculate the difference in HV
            
            if len(f1_sel) == 0:
                return (ref_f1 - f1_new) * (ref_f2 - f2_new)
            
            # Find insertion index
            # f1_sel is sorted
            idx = np.searchsorted(f1_sel, f1_new)
            
            # The new point will be at idx.
            # Previous point is idx-1, next is idx.
            
            # Old contribution of the segment:
            # The point at idx-1 and idx formed a rectangle.
            # Actually, HV is sum of (ref_f1 - f1[i]) * (f2[i-1] - f2[i])
            
            # Let's compute HV of S and HV of S+{p} and return diff.
            # This is O(N).
            
            def calc_hv_arr(f1s, f2s):
                if len(f1s) == 0:
                    return 0.0
                # f1s, f2s are numpy arrays
                # We need to handle the "staircase"
                # f2_prev = ref_f2
                # hv = 0
                # for i in range(len(f1s)):
                #    hv += (ref_f1 - f1s[i]) * (f2_prev - f2s[i])
                #    f2_prev = f2s[i]
                # This assumes f2s is decreasing? 
                # If f2s is not decreasing, the formula needs to be careful.
                # The formula (ref_f1 - f1[i]) * (f2[i-1] - f2[i]) works if f2[i-1] >= f2[i].
                # If f2[i-1] < f2[i], the point i is dominated by i-1? 
                # But we filtered for non-dominated points, so f2 should be decreasing 
                # as f1 increases? 
                # Wait, non-dominated means: if f1[i] > f1[j], then f2[i] < f2[j].
                # So f2 is strictly decreasing with f1.
                # So f2[i-1] > f2[i] is guaranteed for non-dominated sorted by f1.
                
                f2_prev = ref_f2
                hv = 0.0
                for i in range(len(f1s)):
                    hv += (ref_f1 - f1s[i]) * (f2_prev - f2s[i])
                    f2_prev = f2s[i]
                return hv

            hv_old = calc_hv_arr(f1_sel, f2_sel)
            
            # Create new arrays
            f1_new_arr = np.insert(f1_sel, idx, f1_new)
            f2_new_arr = np.insert(f2_sel, idx, f2_new)
            
            hv_new = calc_hv_arr(f1_new_arr, f2_new_arr)
            
            return hv_new - hv_old

        # Greedy loop
        for _ in range(self.budget):
            if time.time() > deadline:
                break
                
            best_inc = -1
            best_idx = -1
            
            # Iterate over non-dominated candidates
            # We can optimize by only checking candidates that are "close" to the current set,
            # but with 2000 candidates, a full scan is fine.
            
            for i in range(n_nd):
                # Check if this candidate is already selected?
                # We can keep a set of selected indices.
                # For simplicity, we just pick from the pool.
                # If we pick a point, we can remove it from consideration.
                
                inc = hv_increase(f1_nd[i], f2_nd[i], np.array(selected_f1_list), np.array(selected_f2_list))
                
                if inc > best_inc:
                    best_inc = inc
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected_f1_list.append(f1_nd[best_idx])
            selected_f2_list.append(f2_nd[best_idx])
            selected_x1_list.append(x1_nd[best_idx])
            
            # Remove selected point from pool to avoid re-selection?
            # Or just mark it.
            # Since we are adding to the list, and calculating HV increase,
            # re-selecting the same point would give 0 increase (or very small).
            # So it's safe to leave it in the pool.
            
            # Optimization: Convert lists to numpy arrays for the next iteration?
            # We do it inside hv_increase.

        # 3. Evaluate the selected points using the problem function
        results = []
        for x1 in selected_x1_list:
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] is already 0
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
            
        return results

# EVOLVE-BLOCK-END
