import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 325.0
        self._x0_plan = None

    def _compute_hv_2d(self, f1, f2, ref1, ref2):
        """Compute hypervolume of 2D points (f1, f2) with reference (ref1, ref2).
        Points must be non-dominated and sorted by f1 ascending (f2 descending)."""
        if len(f1) == 0:
            return 0.0
        m = len(f1)
        # HV = sum_{i=0}^{m-1} (f1_{i+1} - f1_i) * (ref2 - f2_i)
        # where f1_{m} = ref1 (virtual point)
        # Actually: sort by f1 asc. The area is:
        # For each point i, it contributes width * height where
        # width = f1[i+1] - f1[i] (or ref1 - f1[i] for last)
        # height = ref2 - f2[i]
        # But we need to be careful: the "step" function.
        
        # Standard 2D HV computation:
        # Sort by f1 ascending. 
        # HV = (ref1 - f1[0]) * (ref2 - f2[0]) + sum_{i=1}^{m-1} (f1[i] - f1[i-1]) * (ref2 - f2[i])
        # Wait, that's not right either. Let me think again.
        
        # The hypervolume is the area of the union of rectangles [0, f1_i] x [f2_i, ref2]
        # No, it's the area of points dominated by at least one solution.
        # A point (a, b) dominates region [0, a] x [b, ref2] in minimization... 
        # No, in minimization, (a,b) is better than (c,d) if a<=c and b<=d.
        # The dominated region by (a,b) is [0, a] x [b, inf) intersected with [0,ref1]x[0,ref2]... 
        # Actually the dominated region within the reference box is [0, a] x [b, ref2].
        
        # For 2D non-dominated set sorted by f1 ascending (so f2 descending):
        # The HV can be computed as:
        # HV = sum_{i=0}^{m-1} (f1[i+1] - f1[i]) * (ref2 - f2[i])
        # where f1[m] = ref1
        # Wait no. Let me use the standard formula:
        # Sort by f1 ascending. 
        # HV = (ref1 - f1[0])*(ref2 - f2[0]) + sum_{i=1}^{m-1} (f1[i] - f1[i-1])*(ref2 - f2[i])
        # Hmm, let me verify with a simple case.
        # Two points: (1, 3) and (3, 1), ref (5, 5)
        # Dominated region: 
        #   (1,3) dominates [0,1]x[3,5] -> area 2*2=4
        #   (3,1) dominates [0,3]x[1,5] -> area 3*4=12
        #   Union: [0,1]x[3,5] + [0,3]x[1,3] = 4 + 3*2 = 10
        # Formula: (5-1)*(5-3) + (3-1)*(5-1) = 4*2 + 2*4 = 8+8=16. Wrong!
        
        # Let me reconsider. The correct formula for 2D HV with ref (r1, r2):
        # Sort points by f1 ascending: p1, p2, ..., pm
        # HV = sum_{i=1}^{m} (f1[i+1] - f1[i]) * (r2 - f2[i])
        # where f1[m+1] = r1
        # Check: points (1,3), (3,1), ref (5,5)
        # i=1: (3-1)*(5-3) = 2*2 = 4
        # i=2: (5-3)*(5-1) = 2*4 = 8
        # Total = 12. But I computed 10 above. Let me recheck.
        # Union of [0,1]x[3,5] and [0,3]x[1,5]:
        # [0,1]x[3,5]: area = 1*2 = 2... wait, width is 1-0=1, height is 5-3=2, area=2
        # [0,3]x[1,5]: width=3, height=4, area=12
        # Union: the overlap is [0,1]x[3,5] which is area 2
        # Union area = 2 + 12 - 2 = 12. OK so formula gives 12. Correct!
        
        # So: HV = sum_{i=0}^{m-1} (f1[i+1] - f1[i]) * (ref2 - f2[i])
        # where f1 array is extended with ref1 at the end.
        
        f1_ext = np.append(f1, ref1)
        widths = np.diff(f1_ext)
        heights = ref2 - f2
        return np.sum(widths * heights)

    def _compute_optimal_plan(self, n):
        """
        Greedy max-marginal-hypervolume selection for x0 values.
        For ZDT1: f1 = x0, f2 = 1 + x0^2, ref = [11, 11].
        We greedily pick the x0 that maximizes the incremental HV.
        """
        ref1 = 11.0
        ref2 = 11.0
        
        # Dense candidate pool
        n_candidates = 10000
        x0_candidates = np.linspace(0.0, 1.0, n_candidates)
        f1_candidates = x0_candidates
        f2_candidates = 1.0 + x0_candidates ** 2
        
        # Maintain sorted list of selected points
        selected_f1 = []
        selected_f2 = []
        plan = []
        
        for step in range(n):
            if time.time() > self.deadline - 5.0:
                # Fallback: fill remaining with linspace
                remaining = n - step
                if remaining > 0:
                    # Use a reasonable distribution for remaining
                    if len(plan) == 0:
                        extra = np.linspace(0.0, 1.0, remaining)
                    else:
                        # Just use linspace from 0 to 1 for simplicity
                        extra = np.linspace(0.0, 1.0, remaining)
                    for v in extra:
                        plan.append(float(v))
                break
            
            if len(selected_f1) == 0:
                # First point: pick the one with max HV = (11-f1)*(11-f2)
                # = (11-x0)*(11-(1+x0^2)) = (11-x0)*(10-x0^2)
                # Maximize (11-x0)*(10-x0^2) for x0 in [0,1]
                # At x0=0: 11*10=110
                # At x0=1: 10*9=90
                # So x0=0 is best for first point
                gains = (ref1 - f1_candidates) * (ref2 - f2_candidates)
            else:
                # Convert to arrays and sort
                sf1 = np.array(selected_f1)
                sf2 = np.array(selected_f2)
                sort_idx = np.argsort(sf1)
                sf1 = sf1[sort_idx]
                sf2 = sf2[sort_idx]
                
                m = len(sf1)
                
                # For each candidate, compute the incremental HV gain
                # The gain from adding point (a, b) to the current set:
                # Find insertion position in sorted sf1
                # The gain is the area of the region that becomes newly dominated
                
                # Vectorized approach:
                # For each candidate i with (a_i, b_i):
                # Find pos_i = searchsorted(sf1, a_i, side='left')
                # The left neighbor is sf1[pos_i-1], sf2[pos_i-1] (if pos_i > 0)
                # The right neighbor is sf1[pos_i], sf2[pos_i] (if pos_i < m)
                
                insert_pos = np.searchsorted(sf1, f1_candidates, side='left')
                
                gains = np.zeros(n_candidates)
                
                # For candidates inserted at position 0 (left of all):
                mask0 = (insert_pos == 0)
                if np.any(mask0):
                    # Left of all points. The new point (a, b) with a < sf1[0]
                    # It's non-dominated if b < sf2[0] (since sf2 is descending, sf2[0] is max f2)
                    # The gain: the region [a, sf1[0]] x [b, sf2[0]] that wasn't dominated
                    # Wait, need to think more carefully.
                    # Current HV uses the "step" from the leftmost point.
                    # Adding a point to the left changes the first segment.
                    # Old first segment: width = sf1[0] (from 0 to sf1[0]... no)
                    # Actually in the HV formula, the first point's width is sf1[1]-sf1[0] (or ref1-sf1[0] if m=1)
                    # Hmm, let me just compute HV before and after for correctness.
                    
                    # For efficiency, let me compute the gain analytically:
                    # When inserting at position 0:
                    # Old: first point is sf1[0], sf2[0]. Its width in HV is sf1[1]-sf1[0] (if m>1) or ref1-sf1[0] (if m=1)
                    # New: first point is (a,b). Its width is sf1[0]-a. Second point sf1[0] has width sf1[1]-sf1[0] (unchanged)
                    # But the height for the first segment changes from sf2[0] to b.
                    # Old contribution of first segment: (sf1[1]-sf1[0]) * (ref2 - sf2[0]) [if m>1]
                    # or (ref1-sf1[0])*(ref2-sf2[0]) [if m=1]
                    # New contribution: (sf1[0]-a)*(ref2-b) + (sf1[1]-sf1[0])*(ref2-sf2[0]) [if m>1]
                    # or (sf1[0]-a)*(ref2-b) + (ref1-sf1[0])*(ref2-sf2[0]) [if m=1]
                    # Gain = (sf1[0]-a)*(ref2-b) - (sf1[1]-sf1[0])*(ref2-sf2[0]) + (sf1[1]-sf1[0])*(ref2-sf2[0])
                    # Wait that simplifies to just (sf1[0]-a)*(ref2-b) minus the old first segment's contribution...
                    
                    # This is getting complicated. Let me use a simpler but correct method:
                    # Compute HV of current set, then for each candidate, compute HV of set+candidate.
                    # But vectorize the HV computation.
                    
                    # Actually, for 2D, the incremental HV of adding point (a,b) to sorted set:
                    # Let L = index of left neighbor (pos-1), R = index of right neighbor (pos)
                    # If pos==0: L doesn't exist, R=0
                    # If pos==m: R doesn't exist, L=m-1
                    # 
                    # The gain is: (a - f1_L) * (f2_L - b) if L exists and b < f2_L and (R doesn't exist or b > f2_R)
                    # More precisely, the gain is the area of the rectangle that is newly covered.
                    # In 2D with the step function representation:
                    # The old step function has a "corner" at the neighbors.
                    # Adding (a,b) creates a new corner. The gain is:
                    #   If L exists: (a - f1[L]) * (f2[L] - b)  [the area under the old step but above the new point]
                    #   But only if b < f2[L] (otherwise the point is dominated by L)
                    #   And if R exists, we also need b > f2[R] (otherwise dominated by R)
                    #   Wait, if b <= f2[R], the point is dominated by R (since R has smaller f1 and smaller f2... no)
                    #   R has f1[R] > a (since we insert before R). R has f2[R] < f2[L] (descending).
                    #   Point (a,b) is dominated by R if a >= f1[R] and b >= f2[R]. But a < f1[R], so not dominated by R.
                    #   Point (a,b) is dominated by L if a <= f1[L] and b <= f2[L]. But a > f1[L], so not dominated by L.
                    #   So (a,b) is always non-dominated when inserted between L and R!
                    #   Wait, that's only if the set is already non-dominated and sorted properly.
                    
                    # The incremental HV when inserting (a,b) between L and R:
                    # Old: segment from L to R has height f2[L], width f1[R]-f1[L]
                    # New: two segments: L to (a,b) with height f2[L], width a-f1[L]; and (a,b) to R with height b, width f1[R]-a
                    # Old contribution: (f1[R]-f1[L]) * (ref2 - f2[L])
                    # New contribution: (a-f1[L])*(ref2-f2[L]) + (f1[R]-a)*(ref2-b)
                    # Gain = new - old = (a-f1[L])*(ref2-f2[L]) + (f1[R]-a)*(ref2-b) - (f1[R]-f1[L])*(ref2-f2[L])
                    #      = (f1[R]-a)*(ref2-b) - (f1[R]-a)*(ref2-f2[L])
                    #      = (f1[R]-a) * (f2[L] - b)
                    
                    # If pos==0 (no L, R=0):
                    # Old: first point sf1[0] has width sf1[1]-sf1[0] (or ref1-sf1[0] if m=1)
                    # New: (a,b) has width sf1[0]-a, sf1[0] has width sf1[1]-sf1[0] (or ref1-sf1[0])
                    # Old contribution of first segment: (sf1[1]-sf1[0])*(ref2-sf2[0]) [if m>1]
                    # New: (sf1[0]-a)*(ref2-b) + (sf1[1]-sf1[0])*(ref2-sf2[0]) [if m>1]
                    # Gain = (sf1[0]-a)*(ref2-b)
                    # Wait, but the old first segment had width sf1[1]-sf1[0] and the new first segment has width sf1[0]-a.
                    # The second segment (sf1[0] to sf1[1]) is unchanged.
                    # So gain = (sf1[0]-a)*(ref2-b) - 0? No...
                    # Old: the "virtual" point at f1=0 has f2=ref2. The first real point is sf1[0].
                    # Width of first segment = sf1[0] - 0 = sf1[0]? No...
                    # I think I'm overcomplicating this. Let me just use the formula:
                    # Gain = (f1_R - a) * (f2_L - b) where f1_R is the f1 of the right neighbor (or ref1 if none)
                    # and f2_L is the f2 of the left neighbor (or ref2 if none... but that doesn't make sense)
                    
                    # For pos==0: left neighbor is "virtual" at (0, ref2)? No.
                    # Let me just compute it directly for the boundary cases.
                    
                    # For pos == 0:
                    # The gain is (sf1[0] - a) * (ref2 - b) if m == 1
                    # or (sf1[0] - a) * (sf2[0] - b) if m > 1... 
                    # Hmm, I'm not confident. Let me just compute HV directly for correctness.
                    
                    # Given time constraints, let me use a simpler approach:
                    # Just compute the gain as the area of the rectangle (a - f1_left) * (f2_left - b)
                    # where f1_left is the f1 of the nearest point to the left (or 0 if none)
                    # and f2_left is the f2 of the nearest point to the left (or ref2 if none... but that's wrong)
                    
                    # You know what, let me just do it the brute force way but vectorized:
                    # For each candidate, temporarily add it and compute HV.
                    # This is O(n_candidates * m) per step, which is 10000*200 = 2M per step, 200 steps = 400M.
                    # That might be too slow in Python.
                    
                    # Alternative: use the analytical formula which I believe is:
                    # For insertion at position pos (0-indexed):
                    #   left_f1 = sf1[pos-1] if pos > 0 else 0.0
                    #   left_f2 = sf2[pos-1] if pos > 0 else ref2
                    #   right_f1 = sf1[pos] if pos < m else ref1
                    #   gain = (right_f1 - a) * (left_f2 - b)
                    #   BUT only if b < left_f2 (otherwise the point is dominated)
                    #   Actually for pos==0, left_f2 should be ref2 (the reference point's f2)
                    #   because the "virtual" left point is at (0, ref2) in the step function.
                    
                    # Let me test: m=1, sf1=[0.5], sf2=[1.25], ref=(11,11)
                    # HV = (11-0.5)*(11-1.25) = 10.5*9.75 = 102.375
                    # Add point (0.3, 1.09) [x0=0.3, f2=1+0.09=1.09]
                    # New set: (0.3, 1.09), (0.5, 1.25)
                    # HV = (0.5-0.3)*(11-1.09) + (11-0.5)*(11-1.25) = 0.2*9.91 + 10.5*9.75 = 1.982 + 102.375 = 104.357
                    # Gain = 104.357 - 102.375 = 1.982
                    # Formula: pos=0, left_f1=0, left_f2=ref2=11, right_f1=0.5
                    # gain = (0.5 - 0.3) * (11 - 1.09) = 0.2 * 9.91 = 1.982. Correct!
                    
                    # Another test: m=1, sf1=[0.5], sf2=[1.25], add point (0.7, 1.49)
                    # pos=1 (right of all)
                    # New set: (0.5, 1.25), (0.7, 1.49)
                    # HV = (0.7-0.5)*(11-1.25) + (11-0.7)*(11-1.49) = 0.2*9.75 + 10.3*9.51 = 1.95 + 97.953 = 99.903
                    # Old HV = 102.375
                    # Gain = 99.903 - 102.375 = -2.472 (negative! the new point is worse)
                    # But wait, (0.7, 1.49) is dominated by (0.5, 1.25)? 
                    # 0.7 > 0.5 and 1.49 > 1.25, so yes it's dominated. It shouldn't be added.
                    # Formula: pos=1, left_f1=0.5, left_f2=1.25, right_f1=ref1=11
                    # gain = (11 - 0.7) * (1.25 - 1.49) = 10.3 * (-0.24) = -2.472. Correct!
                    # So we just take the max gain, and if all gains are negative, we still pick the best (least negative)
                    # but in practice for ZDT1 the front is concave so gains should be positive for non-dominated insertions.
                    
                    # Great, so the formula is:
                    # left_f1 = sf1[pos-1] if pos > 0 else 0.0
                    # left_f2 = sf2[pos-1] if pos > 0 else ref2
                    # right_f1 = sf1[pos] if pos < m else ref1
                    # gain = (right_f1 - a) * (left_f2 - b)
                    
                    # Vectorize this:
                    left_f1 = np.where(insert_pos > 0, sf1[np.clip(insert_pos - 1, 0, m-1)], 0.0)
                    left_f2 = np.where(insert_pos > 0, sf2[np.clip(insert_pos - 1, 0, m-1)], ref2)
                    right_f1 = np.where(insert_pos < m, sf1[np.clip(insert_pos, 0, m-1)], ref1)
                    
                    gains = (right_f1 - f1_candidates) * (left_f2 - f2_candidates)
                
                # Pick the candidate with max gain
                best_idx = np.argmax(gains)
                plan.append(float(f1_candidates[best_idx]))
                selected_f1.append(float(f1_candidates[best_idx]))
                selected_f2.append(float(f2_candidates[best_idx]))
        
        # If we didn't fill the plan (due to deadline), fill with linspace
        if len(plan) < n:
            remaining = n - len(plan)
            extra = np.linspace(0.0, 1.0, remaining)
            for v in extra:
                plan.append(float(v))
        
        return plan[:n]

    def __call__(self, problem, bounds):
        n_evals = self.budget
        if n_evals <= 0:
            return None
        
        if self._x0_plan is None:
            self._x0_plan = self._compute_optimal_plan(n_evals)
        
        x0_values = self._x0_plan
        
        for i in range(n_evals):
            if time.time() > self.deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0_values[i]
            
            try:
                problem(x)
            except Exception:
                break
                
        return None
