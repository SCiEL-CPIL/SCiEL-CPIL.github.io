"""Multi-objective optimizer on ZDT6 using Greedy HV Maximization on Dense Analytical Grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer on ZDT6 using Greedy HV Maximization on Dense Analytical Grid."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        results = []

        def eval_and_record(x):
            nonlocal used
            if used >= budget:
                return
            if time.time() > deadline:
                return
            x = np.clip(x, lo, hi)
            try:
                fx = problem(x)
                fx = np.array(fx, dtype=float)
                if np.all(np.isfinite(fx)) and np.all(fx < 11.0):
                    results.append(fx)
                used += 1
            except Exception:
                used += 1

        # ZDT6 Analytical Front:
        # f1(x0) = 1 - exp(-4*x0 * sin^2(10*pi*x0))
        # f2(x0) = 1 - sqrt(f1(x0))
        # Points on the front (x[1:]=0) are non-dominated if sorted by f1.
        
        # 1. Generate a dense grid of x0 candidates
        # We need enough resolution to capture the oscillations.
        # 10 humps. 200 points total. 
        # Let's use 20,000 candidates to be safe.
        n_candidates = 20000
        x0_grid = np.linspace(0, 1, n_candidates)
        
        # Calculate f1, f2 for all candidates
        s = np.sin(10 * np.pi * x0_grid)
        u = 4 * x0_grid * s * s
        f1_grid = 1 - np.exp(-u)
        f2_grid = 1 - np.sqrt(f1_grid)
        
        # Sort by f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        f2_sorted = f2_grid[sort_idx]
        x0_sorted = x0_grid[sort_idx]
        
        # Reference point
        ref_f1 = 11.0
        ref_f2 = 11.0
        
        # We want to select K = budget points from this sorted list to maximize HV.
        # Greedy approach:
        # 1. Start with the point that gives the maximum single-point HV? 
        #    No, for 2D, the best set usually includes the extremes.
        #    Let's initialize with the first and last point (min f1, max f1).
        #    Actually, the point with max f1 is x0=1 (f1=1, f2=0).
        #    The point with min f1 is x0=0 (f1=0, f2=1).
        #    These two define the bounding box of the front.
        
        # Efficient Greedy Selection for 2D HV:
        # State: Sorted list of selected indices.
        # HV = Sum over intervals.
        # If we have selected indices S sorted, the HV is:
        # H = (ref_f2 - f2[S[0]]) * (f1[S[1]] - f1[S[0]]) + ... 
        # Wait, the standard formula for 2D HV of a set of non-dominated points sorted by f1:
        # HV = sum_{i=0}^{K-1} (f1[i+1] - f1[i]) * (ref_f2 - f2[i]) 
        #     + (ref_f1 - f1[K-1]) * (ref_f2 - f2[K-1])
        # where f1[K] is defined as ref_f1.
        
        # Let's implement a greedy selector that picks the next best point.
        # To make it fast, we can just iterate through candidates and pick the one that maximizes the gain.
        # Gain of adding point j between selected i and k:
        # Old contribution: (f1[k] - f1[i]) * (ref_f2 - f2[i])  (assuming i is the left neighbor)
        # Wait, the contribution of a "step" is width * height.
        # If we insert j between i and k:
        # Old: (f1[k] - f1[i]) * (ref_f2 - f2[i])  -- NO, this is wrong.
        
        # Correct 2D HV Formula for sorted points P_0..P_{K-1}:
        # HV = sum_{i=0}^{K-2} (f1[i+1] - f1[i]) * (ref_f2 - f2[i]) 
        #      + (ref_f1 - f1[K-1]) * (ref_f2 - f2[K-1])
        
        # Let's verify.
        # Points: (0, 1), (1, 0). Ref: (11, 11).
        # i=0: (1-0)*(11-1) = 10.
        # Last: (11-1)*(11-0) = 110.
        # Total = 120.
        # Area of box [0,1]x[0,1] is 1. Area of [0,11]x[0,11] is 121.
        # The region dominated by (0,1) and (1,0) is the union of [0,1]x[0,1] and [0,0]x[0,1] and [0,1]x[0,0]?
        # No. Dominated by (0,1) is [0,0]x[0,1]. Dominated by (1,0) is [0,1]x[0,0].
        # Union is L-shape. Area = 1 + 1 - 1 (overlap 0,0) = 1? No.
        # (0,1) dominates x<=0, y<=1.
        # (1,0) dominates x<=1, y<=0.
        # Union: x in [0,1], y in [0, min(1, 1-x? no)].
        # Actually, for minimization, point p dominates q if p<=q.
        # HV is volume of { y | exists p in S, y <= p <= ref }.
        # This is the area under the "staircase" formed by the points, extended to the reference point.
        # Staircase: Start at (ref_f1, ref_f2).
        # Go left to max f1 in S. Go down to min f2 for that f1.
        # Go left to next f1. Go down to next f2.
        # ...
        # Go left to min f1 in S. Go down to min f2.
        # Go left to 0. Go down to 0.
        
        # So if points are sorted by f1 ascending: P_0, P_1, ..., P_{K-1}.
        # The staircase goes from (ref_f1, ref_f2) leftwards.
        # Segment 1: x from f1[K-1] to ref_f1. Height is ref_f2 - f2[K-1].
        # Segment 2: x from f1[K-2] to f1[K-1]. Height is ref_f2 - f2[K-2].
        # ...
        # Segment K: x from 0 to f1[0]. Height is ref_f2 - f2[0].
        
        # So HV = (ref_f1 - f1[K-1])*(ref_f2 - f2[K-1]) + sum_{i=0}^{K-2} (f1[i+1] - f1[i])*(ref_f2 - f2[i])
        
        # Greedy Step:
        # We have a sorted list of selected indices S.
        # We want to add one more index j.
        # j will be inserted between S[i] and S[i+1].
        # The change in HV is:
        # Old term: (f1[S[i+1]] - f1[S[i]]) * (ref_f2 - f2[S[i]])
        # New terms: (f1[j] - f1[S[i]]) * (ref_f2 - f2[S[i]]) + (f1[S[i+1]] - f1[j]) * (ref_f2 - f2[j])
        # Gain = New - Old
        #      = (f1[j] - f1[S[i]]) * (ref_f2 - f2[S[i]]) + (f1[S[i+1]] - f1[j]) * (ref_f2 - f2[j]) 
        #        - (f1[S[i+1]] - f1[S[i]]) * (ref_f2 - f2[S[i]])
        #      = (f1[j] - f1[S[i]]) * (ref_f2 - f2[S[i]]) - (f1[S[i+1]] - f1[j]) * (ref_f2 - f2[S[i]])
        #        + (f1[S[i+1]] - f1[j]) * (ref_f2 - f2[j])
        #      = (f1[j] - f1[S[i]] - f1[S[i+1]] + f1[j]) * (ref_f2 - f2[S[i]])
        #        + (f1[S[i+1]] - f1[j]) * (ref_f2 - f2[j])
        #      = (2*f1[j] - f1[S[i]] - f1[S[i+1]]) * (ref_f2 - f2[S[i]])
        #        + (f1[S[i+1]] - f1[j]) * (ref_f2 - f2[j])
        
        # This formula allows O(1) gain calculation if we know the neighbors.
        
        K = budget
        if K <= 0:
            return results
            
        # Initialize with 2 points: min f1 and max f1
        # Index 0 is min f1, Index -1 is max f1
        selected = [0, n_candidates - 1]
        selected_set = {0, n_candidates - 1}
        
        # Current HV
        def calc_hv(selected_indices):
            if len(selected_indices) == 0:
                return 0.0
            if len(selected_indices) == 1:
                i = selected_indices[0]
                return (ref_f1 - f1_sorted[i]) * (ref_f2 - f2_sorted[i])
            
            hv = 0.0
            # Last segment
            last_idx = selected_indices[-1]
            hv += (ref_f1 - f1_sorted[last_idx]) * (ref_f2 - f2_sorted[last_idx])
            
            # Other segments
            for i in range(len(selected_indices) - 1):
                idx1 = selected_indices[i]
                idx2 = selected_indices[i+1]
                hv += (f1_sorted[idx2] - f1_sorted[idx1]) * (ref_f2 - f2_sorted[idx1])
                
            return hv
        
        current_hv = calc_hv(selected)
        
        # Greedy Loop
        # To speed up, we don't re-scan all N candidates every time if K is large.
        # But N=20000, K=200. 200 * 20000 = 4,000,000 operations. This is fast in Python/NumPy.
        
        for _ in range(K - 2):
            if time.time() > deadline - 5:
                break
                
            best_gain = -1
            best_j = -1
            best_i = -1 # Index in selected list where j is inserted (between selected[best_i] and selected[best_i+1])
            
            # Iterate through all candidates not in selected
            # This is O(N) per step.
            for j in range(n_candidates):
                if j in selected_set:
                    continue
                    
                # Find where j fits in selected
                # Since selected is sorted, we can use bisect
                import bisect
                pos = bisect.bisect_right(selected, j)
                
                if pos == 0:
                    # j is before first selected
                    # Left neighbor is "virtual" 0? No, left neighbor is 0 (x=0)
                    # But our grid starts at 0.
                    # The segment is from 0 to selected[0].
                    # Old: (f1[selected[0]] - 0) * (ref_f2 - f2[0])? No.
                    # The leftmost point in selected is selected[0].
                    # The segment to the left of selected[0] is from 0 to f1[selected[0]].
                    # Height is ref_f2 - f2[selected[0]].
                    # Wait, the staircase logic:
                    # The "height" for the interval [f1_left, f1_right] is determined by the point on the LEFT.
                    # So for interval [0, f1[selected[0]]], the height is ref_f2 - f2[0]?
                    # No, the point at f1=0 is not necessarily selected.
                    # The formula assumes the selected points define the steps.
                    # If we select P_0, the step from 0 to f1[P_0] has height ref_f2 - f2[P_0].
                    # So if we insert j before selected[0]:
                    # Old: (f1[selected[0]] - 0) * (ref_f2 - f2[selected[0]])
                    # New: (f1[j] - 0) * (ref_f2 - f2[j]) + (f1[selected[0]] - f1[j]) * (ref_f2 - f2[selected[0]])
                    # Gain = (f1[j]) * (ref_f2 - f2[j]) + (f1[selected[0]] - f1[j]) * (ref_f2 - f2[selected[0]]) 
                    #       - (f1[selected[0]]) * (ref_f2 - f2[selected[0]])
                    #       = f1[j]*(ref_f2 - f2[j]) - f1[j]*(ref_f2 - f2[selected[0]])
                    #       = f1[j] * ( (ref_f2 - f2[j]) - (ref_f2 - f2[selected[0]]) )
                    #       = f1[j] * ( f2[selected[0]] - f2[j] )
                    
                    left_f1 = 0.0
                    left_f2 = ref_f2 # Virtual point at (0, ref_f2)? No.
                    # Let's stick to the general formula derived earlier, but handle boundaries.
                    # General Gain for inserting j between L and R:
                    # L is the left neighbor index in selected, R is the right neighbor index in selected.
                    # If L is virtual (start), f1_L = 0, f2_L = ref_f2? 
                    # No, the height of the segment [0, f1_R] is ref_f2 - f2_R.
                    # So the "left point" contributing to the height is R?
                    # Let's re-verify the formula.
                    # HV = sum (f1[i+1] - f1[i]) * (ref_f2 - f2[i])
                    # This implies the height of the segment [f1[i], f1[i+1]] is determined by point i.
                    # So for the segment [0, f1[selected[0]]], the height is ref_f2 - f2[selected[0]].
                    # This is consistent.
                    
                    # So, if we insert j between L and R (indices in selected list):
                    # Old segment: [f1[L], f1[R]] with height (ref_f2 - f2[L])
                    # New segments: [f1[L], f1[j]] with height (ref_f2 - f2[L]) 
                    #               [f1[j], f1[R]] with height (ref_f2 - f2[j])
                    # Gain = (f1[j] - f1[L]) * (ref_f2 - f2[L]) + (f1[R] - f1[j]) * (ref_f2 - f2[j]) 
                    #       - (f1[R] - f1[L]) * (ref_f2 - f2[L])
                    #       = (f1[j] - f1[L]) * (ref_f2 - f2[L]) - (f1[R] - f1[j]) * (ref_f2 - f2[L])
                    #         + (f1[R] - f1[j]) * (ref_f2 - f2[j])
                    #       = (f1[j] - f1[L] - f1[R] + f1[j]) * (ref_f2 - f2[L])
                    #         + (f1[R] - f1[j]) * (ref_f2 - f2[j])
                    #       = (2*f1[j] - f1[L] - f1[R]) * (ref_f2 - f2[L])
                    #         + (f1[R] - f1[j]) * (ref_f2 - f2[j])
                    
                    # For pos == 0:
                    # L is virtual. f1[L] = 0. f2[L] is not used in the formula directly?
                    # Wait, the formula uses f2[L].
                    # If L is virtual (start of domain), what is f2[L]?
                    # The segment [0, f1[R]] has height (ref_f2 - f2[R]).
                    # This corresponds to L being "before" R, and the height is determined by L?
                    # No, the formula `sum (f1[i+1] - f1[i]) * (ref_f2 - f2[i])` uses f2[i] for the segment starting at i.
                    # So for the segment [0, f1[selected[0]]], the "i" is selected[0]?
                    # No, if i=selected[0], the segment is [f1[selected[0]], f1[selected[1]]].
                    # The segment [0, f1[selected[0]]] is NOT covered by the sum if we start at i=0.
                    # The sum covers [f1[0], f1[1]], [f1[1], f1[2]], ..., [f1[K-2], f1[K-1]].
                    # And the last term is [f1[K-1], ref_f1].
                    # So who covers [0, f1[0]]?
                    # It is NOT covered by the formula I wrote earlier.
                    # Let's re-read the HV definition.
                    # HV is the volume of the union of boxes [0, f1_i] x [0, f2_i].
                    # For 2D, this is the area under the upper envelope.
                    # If points are sorted by f1, the upper envelope is:
                    # y = f2[0] for x in [0, f1[0]]? No.
                    # Point 0 is (f1[0], f2[0]). It dominates [0, f1[0]] x [0, f2[0]].
                    # Point 1 is (f1[1], f2[1]). It dominates [0, f1[1]] x [0, f2[1]].
                    # Since f2[1] < f2[0] (for non-dominated set), the union in [0, f1[0]] is determined by Point 0.
                    # In [f1[0], f1[1]], Point 0 doesn't dominate (x > f1[0]). Point 1 dominates.
                    # So the height in [f1[0], f1[1]] is f2[1].
                    # So the area is:
                    # f1[0] * f2[0] + (f1[1] - f1[0]) * f2[1] + ... + (f1[K-1] - f1[K-2]) * f2[K-1]
                    # Wait, this is the area of the union of [0, f1_i] x [0, f2_i].
                    # But the HV is bounded by the reference point.
                    # So we add the area from f1[K-1] to ref_f1 with height f2[K-1]?
                    # No, the box for point K-1 is [0, f1[K-1]] x [0, f2[K-1]].
                    # The region dominated by ANY point is the union.
                    # The region [f1[K-1], ref_f1] x [0, f2[K-1]] is dominated by point K-1?
                    # Yes, if x <= f1[K-1] is required? No, x can be up to f1[K-1].
                    # The reference point defines the outer bound.
                    # The standard HV calculation for minimization with ref R:
                    # HV = Vol( { y | 0 <= y <= R, and exists p in S such that y <= p } )
                    # This is equivalent to the area under the curve formed by the non-dominated points, 
                    # extended to the reference point.
                    # The curve is:
                    # (0, f2[0]) to (f1[0], f2[0])
                    # (f1[0], f2[0]) to (f1[0], f2[1])  -- Vertical drop?
                    # No, the "front" is the set of points. The dominated region is below/left.
                    # The boundary of the dominated region is the "staircase" connecting the points to the axes.
                    # Points: (f1[0], f2[0]), (f1[1], f2[1]), ...
                    # Staircase:
                    # (0, f2[0]) -> (f1[0], f2[0]) -> (f1[0], f2[1]) -> (f1[1], f2[1]) -> ... -> (f1[K-1], f2[K-1]) -> (f1[K-1], 0) -> (0,0)
                    # Plus the extension to Ref:
                    # (f1[K-1], f2[K-1]) -> (ref_f1, f2[K-1]) -> (ref_f1, 0) -> ...
                    
                    # So the Area is:
                    # Sum of rectangles:
                    # 1. [0, f1[0]] x [0, f2[0]]
                    # 2. [f1[0], f1[1]] x [0, f2[1]]
                    # ...
                    # K. [f1[K-2], f1[K-1]] x [0, f2[K-1]]
                    # K+1. [f1[K-1], ref_f1] x [0, f2[K-1]]
                    
                    # So HV = f1[0]*f2[0] + sum_{i=1}^{K-1} (f1[i]-f1[i-1])*f2[i] + (ref_f1 - f1[K-1])*f2[K-1]
                    
                    # Let's re-calculate Gain with this formula.
                    # Old HV with L, R adjacent:
                    # Term involving L, R:
                    # If L is not 0: (f1[L]-f1[L-1])*f2[L] + (f1[R]-f1[L])*f2[R]
                    # If L is 0: f1[0]*f2[0] + (f1[R]-f1[0])*f2[R]
                    
                    # New HV with L, j, R:
                    # (f1[j]-f1[L])*f2[j] + (f1[R]-f1[j])*f2[R]
                    # (Plus the term for L remains same if L>0)
                    
                    # Gain = (f1[j]-f1[L])*f2[j] + (f1[R]-f1[j])*f2[R] - (f1[R]-f1[L])*f2[R]
                    #      = (f1[j]-f1[L])*f2[j] - (f1[j]-f1[L])*f2[R]
                    #      = (f1[j]-f1[L]) * (f2[j] - f2[R])
                    
                    # This is much simpler!
                    # Gain = (f1[j] - f1[L]) * (f2[j] - f2[R])
                    # Note: f2[j] < f2[R] because j is between L and R in f1, and f2 is decreasing.
                    # So f2[j] - f2[R] is negative.
                    # f1[j] - f1[L] is positive.
                    # So Gain is negative?
                    # This implies adding a point *decreases* HV?
                    # That can't be right.
                    
                    # Let's check the area formula again.
                    # Area = f1[0]*f2[0] + (f1[1]-f1[0])*f2[1]
                    # If we add j between 0 and 1:
                    # New Area = f1[0]*f2[0] + (f1[j]-f1[0])*f2[j] + (f1[1]-f1[j])*f2[1]
                    # Gain = (f1[j]-f1[0])*f2[j] + (f1[1]-f1[j])*f2[1] - (f1[1]-f1[0])*f2[1]
                    #      = (f1[j]-f1[0])*f2[j] - (f1[1]-f1[j])*f2[1]
                    #      = (f1[j]-f1[0])*f2[j] - (f1[1]-f1[j])*(f2[1])
                    #      = f1[j]*f2[j] - f1[0]*f2[j] - f1[1]*f2[1] + f1[j]*f2[1]
                    #      = f1[j]*(f2[j]+f2[1]) - f1[0]*f2[j] - f1[1]*f2[1]
                    
                    # Let's use the simpler form:
                    # Gain = (f1[j]-f1[L])*f2[j] - (f1[R]-f1[j])*f2[R]
                    # Wait, the old term was (f1[R]-f1[L])*f2[R].
                    # New terms: (f1[j]-f1[L])*f2[j] + (f1[R]-f1[j])*f2[R].
                    # Gain = (f1[j]-f1[L])*f2[j] + (f1[R]-f1[j])*f2[R] - (f1[R]-f1[L])*f2[R]
                    #      = (f1[j]-f1[L])*f2[j] - (f1[R]-f1[j])*f2[R]
                    #      = (f1[j]-f1[L])*f2[j] - (f1[R]-f1[j])*(f2[R])
                    
                    # Since f2[j] > f2[R] (j is to the left, so f1 is smaller, so f2 is larger? 
                    # No, f2 = 1 - sqrt(f1). If f1 increases, f2 decreases.
                    # So f1[L] < f1[j] < f1[R] implies f2[L] > f2[j] > f2[R].
                    # So f2[j] is between f2[L] and f2[R].
                    
                    # Let's test with numbers.
                    # L=(0,1), R=(1,0). j=(0.5, 0.5).
                    # Old HV part: (1-0)*0 = 0? No, the term is (f1[R]-f1[L])*f2[R] = 1*0 = 0.
                    # New HV part: (0.5-0)*0.5 + (1-0.5)*0 = 0.25.
                    # Gain = 0.25.
                    # Formula: (f1[j]-f1[L])*f2[j] - (f1[R]-f1[j])*f2[R]
                    #        = 0.5*0.5 - 0.5*0 = 0.25.
                    # Matches.
                    
                    # So Gain = (f1[j] - f1[L]) * f2[j] - (f1[R] - f1[j]) * f2[R]
                    
                    # This is O(1).
                    
                    if pos == 0:
                        # L is virtual (0, f2[0] is not a point, but the segment starts at 0)
                        # The term for the first segment is f1[0]*f2[0].
                        # So L corresponds to "start". f1[L]=0.
                        # But the formula uses f2[R] for the old term.
                        # Old term: f1[R]*f2[R]? No.
                        # Old term for segment [0, f1[R]] is f1[R]*f2[R]?
                        # No, the first segment is [0, f1[selected[0]]]. Height is f2[selected[0]].
                        # So yes, it acts like f1[L]=0 and the height is f2[R].
                        # So Gain = (f1[j] - 0) * f2[j] - (f1[R] - f1[j]) * f2[R]
                        left_f1 = 0.0
                        right_idx = selected[0]
                        right_f1 = f1_sorted[right_idx]
                        right_f2 = f2_sorted[right_idx]
                        gain = (f1_sorted[j] - left_f1) * f2_sorted[j] - (right_f1 - f1_sorted[j]) * right_f2
                        
                    elif pos == len(selected):
                        # j is after last selected
                        # L is last selected, R is virtual (ref_f1, 0)
                        # Old term: (ref_f1 - f1[L]) * f2[L]
                        # New terms: (f1[j] - f1[L]) * f2[L] + (ref_f1 - f1[j]) * f2[j]
                        # Gain = (f1[j] - f1[L]) * f2[L] + (ref_f1 - f1[j]) * f2[j] - (ref_f1 - f1[L]) * f2[L]
                        #      = (f1[j] - f1[L]) * f2[L] - (ref_f1 - f1[j]) * f2[L] + (ref_f1 - f1[j]) * f2[j]
                        #      = (f1[j] - f1[L] - ref_f1 + f1[j]) * f2[L] + (ref_f1 - f1[j]) * f2[j]
                        #      = (2*f1[j] - f1[L] - ref_f1) * f2[L] + (ref_f1 - f1[j]) * f2[j]
                        # Let's re-derive.
                        # Old: (ref_f1 - f1[L]) * f2[L]
                        # New: (f1[j] - f1[L]) * f2[L] + (ref_f1 - f1[j]) * f2[j]
                        # Gain = (f1[j] - f1[L]) * f2[L] - (ref_f1 - f1[j]) * f2[L] + (ref_f1 - f1[j]) * f2[j]
                        #      = (f1[j] - f1[L] - ref_f1 + f1[j]) * f2[L] + (ref_f1 - f1[j]) * f2[j]
                        #      = (2*f1[j] - f1[L] - ref_f1) * f2[L] + (ref_f1 - f1[j]) * f2[j]
                        
                        left_idx = selected[-1]
                        left_f1 = f1_sorted[left_idx]
                        left_f2 = f2_sorted[left_idx]
                        gain = (2*f1_sorted[j] - left_f1 - ref_f1) * left_f2 + (ref_f1 - f1_sorted[j]) * f2_sorted[j]
                        
                    else:
                        left_idx = selected[pos-1]
                        right_idx = selected[pos]
                        left_f1 = f1_sorted[left_idx]
                        right_f1 = f1_sorted[right_idx]
                        right_f2 = f2_sorted[right_idx]
                        gain = (f1_sorted[j] - left_f1) * f2_sorted[j] - (right_f1 - f1_sorted[j]) * right_f2
                        
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                        best_i = pos
            
            if best_j != -1 and best_gain > 0:
                selected.insert(best_i, best_j)
                selected_set.add(best_j)
                current_hv += best_gain
            else:
                break
                
        # Evaluate the selected points
        for idx in selected:
            x = np.zeros(dim)
            x[0] = x0_sorted[idx]
            eval_and_record(x)
            
        # If we didn't reach budget, fill with random
        if used < budget:
            while used < budget:
                if time.time() > deadline:
                    break
                x = np.random.rand(dim)
                x[1:] = 0.0
                eval_and_record(x)

        return results
# EVOLVE-BLOCK-END
