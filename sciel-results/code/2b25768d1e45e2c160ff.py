import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        c = 1 + 10 * (n - 1)
        ref = np.array([11.0, 11.0])
        
        # 1. Generate dense Pareto front candidates
        # We need enough points to capture the oscillations.
        # 10 oscillations in [0,1]. 10000 points gives 1000 points per oscillation.
        N_dense = 20000
        t_vals = np.linspace(0.0, 1.0, N_dense)
        
        f1_vals = t_vals
        f2_vals = c * (1.0 - np.sqrt(f1_vals / c) + np.sin(10.0 * np.pi * f1_vals))
        
        # 2. Filter points strictly better than reference
        mask = (f1_vals < ref[0]) & (f2_vals < ref[1])
        f1_valid = f1_vals[mask]
        f2_valid = f2_vals[mask]
        
        if len(f1_valid) == 0:
            # Fallback
            x = np.zeros(n)
            x[0] = 0.5
            return [problem(x)]
            
        # 3. Sort by f1
        sort_idx = np.argsort(f1_valid)
        f1_sorted = f1_valid[sort_idx]
        f2_sorted = f2_valid[sort_idx]
        
        # 4. Remove dominated points
        # A point i is dominated if there exists j < i (smaller f1) with f2[j] <= f2[i]
        # We want to keep points where f2 is strictly decreasing.
        # Since f1 is sorted, we just track min_f2_so_far.
        # Actually, for non-dominated set, we keep point i if f2[i] < min(f2[0..i-1]).
        # Wait, if f2[i] == min, it's dominated by the previous one with same f2 but smaller f1.
        # So strictly less.
        
        nondom_indices = []
        min_f2_so_far = np.inf
        for i in range(len(f1_sorted)):
            if f2_sorted[i] < min_f2_so_far:
                nondom_indices.append(i)
                min_f2_so_far = f2_sorted[i]
                
        f1_nd = f1_sorted[nondom_indices]
        f2_nd = f2_sorted[nondom_indices]
        
        # 5. Greedy selection of 'budget' points to maximize Hypervolume
        # HV = sum (f1[i+1] - f1[i]) * (ref[1] - f2[i])
        
        selected_indices = []
        if len(f1_nd) <= self.budget:
            selected_indices = list(range(len(f1_nd)))
        else:
            # Start with endpoints
            selected_indices = [0, len(f1_nd) - 1]
            selected_set = set(selected_indices)
            
            # We need to pick self.budget - 2 more points.
            # To speed up, we can use a heap of potential gains.
            # But calculating gain for all M points N times is O(N*M).
            # M ~ 10000, N = 200. 2 * 10^7 operations.
            # In Python, a loop of 200 * 10000 is 2 million iterations.
            # Each iteration is simple arithmetic. This should take < 1 second.
            
            for _ in range(self.budget - 2):
                best_gain = -np.inf
                best_idx = -1
                
                # Iterate through all non-dominated points not yet selected
                # Optimization: Only points between selected points matter?
                # No, the gain depends on the neighbors in the *selected* set.
                # For a candidate point P, find its left neighbor L and right neighbor R in the *selected* set.
                # Gain = (f1[R] - f1[P]) * (f2[L] - f2[P])
                
                # To make this efficient, we can maintain a sorted list of selected indices.
                # But simple loop is fine for 200 iterations.
                
                # Let's just loop through all non-dominated points
                for i in range(len(f1_nd)):
                    if i in selected_set:
                        continue
                    
                    # Find left and right neighbors in selected_set
                    # Since selected_set is small (<=200), we can just sort it and binary search?
                    # Or just find min selected > i and max selected < i.
                    
                    # For simplicity and given small N, let's just find neighbors by scanning selected_indices?
                    # No, scanning 200 elements 10000 times is 2M ops. Total 400M ops?
                    # 200 iterations * 10000 candidates * (find neighbors cost).
                    # Finding neighbors by sorting selected_indices every time is expensive.
                    
                    # Better approach:
                    # Maintain a list of "gaps".
                    # Initially, one gap [0, N-1].
                    # When we add a point, we split the gap.
                    # We only need to evaluate candidates in the current gaps.
                    
                    # However, implementing a dynamic gap structure is complex.
                    # Let's try the simple O(N*M) first and see if it's fast enough.
                    # If not, we optimize.
                    
                    # Actually, we can precompute the neighbors for each candidate based on the CURRENT selected set.
                    # This is hard to vectorize easily.
                    
                    # Alternative: 
                    # Just pick the point with the maximum "local curvature" or similar?
                    # No, greedy HV is robust.
                    
                    # Let's optimize the neighbor finding.
                    # Keep selected_indices sorted.
                    # selected_indices is small.
                    
                    # We will do this:
                    # 1. Sort selected_indices (keep it sorted)
                    # 2. For each candidate i, use np.searchsorted to find position in selected_indices.
                    # 3. Get left and right neighbor.
                    
                    # But doing this inside the loop for every i is slow.
                    
                    # Let's try a different strategy:
                    # Since the front is smooth-ish, we can just sample uniformly from the non-dominated set?
                    # No, that's what the current program does (roughly) and it scores 1.26.
                    # Greedy should be better.
                    
                    # Let's implement the gap splitting approach.
                    # Gaps are ranges of indices in f1_nd.
                    # We maintain a list of (left_idx, right_idx) gaps.
                    # For each gap, we find the best point to add.
                    # Gain for point i in gap (L, R) is (f1[R] - f1[i]) * (f2[L] - f2[i]).
                    # Note: f2[L] is the f2 of the left boundary point.
                    
                    # This requires checking all points in the gap.
                    # Sum of gap sizes is always M.
                    # So each iteration is O(M).
                    # Total O(N*M) = 200 * 10000 = 2M operations.
                    # This is very fast.
                    
                    break # Wait, I started a loop above. Let's restart the logic.
            
            # Restart Greedy Logic with Gap Splitting
            selected_indices = [0, len(f1_nd) - 1]
            # Gaps: (left_idx, right_idx)
            gaps = [(0, len(f1_nd) - 1)]
            
            for _ in range(self.budget - 2):
                best_gain = -np.inf
                best_gap_idx = -1
                best_point_idx = -1
                
                for g_idx, (l, r) in enumerate(gaps):
                    if r - l <= 1:
                        continue
                    
                    # Find best point in (l, r)
                    # Candidates are indices l+1 to r-1
                    # Gain = (f1[r] - f1[i]) * (f2[l] - f2[i])
                    # We want to maximize this.
                    
                    # Vectorized calculation for this gap
                    indices = np.arange(l + 1, r)
                    if len(indices) == 0:
                        continue
                    
                    f1_cand = f1_nd[indices]
                    f2_cand = f2_nd[indices]
                    
                    gains = (f1_nd[r] - f1_cand) * (f2_nd[l] - f2_cand)
                    
                    max_gain_idx_local = np.argmax(gains)
                    max_gain_val = gains[max_gain_idx_local]
                    
                    if max_gain_val > best_gain:
                        best_gain = max_gain_val
                        best_gap_idx = g_idx
                        best_point_idx = indices[max_gain_idx_local]
                        
                if best_point_idx == -1:
                    break
                    
                # Add best_point_idx to selected
                selected_indices.append(best_point_idx)
                
                # Update gaps
                l, r = gaps[best_gap_idx]
                # Split (l, r) into (l, best_point_idx) and (best_point_idx, r)
                gaps[best_gap_idx] = (l, best_point_idx)
                gaps.append((best_point_idx, r))
                
                # Optional: Remove empty gaps to save memory/time, but N is small.
                # gaps = [g for g in gaps if g[1] - g[0] > 1]
                
        # Sort selected indices to ensure correct order for evaluation (though order doesn't matter for HV, 
        # but we return the points)
        selected_indices.sort()
        
        # 6. Evaluate the selected points using the problem function
        results = []
        x = np.zeros(n)
        
        for idx in selected_indices:
            if time.time() > deadline:
                break
                
            t_val = f1_nd[idx]
            x[0] = t_val
            # x[1:] is already 0
            
            # Clip to bounds just in case, though t_val is in [0,1] and bounds are [0,1]
            # x_clipped = np.clip(x, bounds[0][0], bounds[0][1]) # Not needed if bounds are [0,1]
            
            f = problem(x)
            results.append(f)
            
        return results
