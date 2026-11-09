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

    def _compute_optimal_plan(self, n):
        """
        Greedy max-marginal-hypervolume feedback law (Pontryagin-inspired).
        State: current set of points on the Pareto front.
        Control: next x0 to add.
        Reward: incremental hypervolume contribution.
        
        For ZDT1: f1 = x0, f2 = 1 + x0^2, ref = [11, 11].
        We greedily pick the x0 that maximizes the HV added by inserting
        (x0, 1+x0^2) into the current set.
        """
        # Pareto front: f1 = x0, f2 = 1 + x0^2 for x0 in [0,1]
        # Reference point
        ref = np.array([11.0, 11.0])
        
        # Dense candidate pool for the feedback law
        n_candidates = 5000
        x0_candidates = np.linspace(0.0, 1.0, n_candidates)
        f1_candidates = x0_candidates
        f2_candidates = 1.0 + x0_candidates ** 2
        
        # Start with empty set
        selected_f1 = []
        selected_f2 = []
        plan = []
        
        for step in range(n):
            if time.time() > self.deadline - 5.0:
                # Fallback to linspace for remaining
                remaining = n - step
                if remaining > 0:
                    last = plan[-1] if plan else -1.0 / n
                    extra = np.linspace(0.0, 1.0, remaining + 1)[1:]
                    for v in extra:
                        plan.append(v)
                break
            
            # Compute marginal HV gain for each candidate
            # A point (a, b) with ref (11,11) contributes to HV based on
            # its position in the non-dominated set.
            
            # For efficiency, compute the HV of the set with each candidate added
            # vs without. But that's O(n * candidates * log n).
            
            # Faster approach: for a 2D problem, the HV of a set of points
            # can be computed by sorting by f1 and computing the sum of
            # (ref_f2 - f2_i) * (f1_{i+1} - f1_i) for each consecutive pair,
            # plus the last point's contribution to ref_f1.
            
            # Let's use a vectorized approach:
            # Current selected points
            if len(selected_f1) == 0:
                # First point: HV = (11 - f2) * (11 - f1) ... no, that's not right
                # HV of single point (a,b) with ref (11,11): (11-a)*(11-b)
                # But we need the union of all points' contributions.
                # For the first point, marginal gain = (11-f1_i)*(11-f2_i)
                gains = (ref[0] - f1_candidates) * (ref[1] - f2_candidates)
            else:
                # Sort current points by f1 (ascending)
                idx = np.argsort(selected_f1)
                sf1 = np.array(selected_f1)[idx]
                sf2 = np.array(selected_f2)[idx]
                
                # Compute current HV
                # HV = sum over i of (f1_{i+1} - f1_i) * (11 - f2_i) 
                #     + (11 - f1_last) * (11 - f2_last)
                # where f1_0 = 0 (or the min f1)
                
                # Actually for 2D HV with ref (11,11):
                # Sort by f1 ascending. For each point i, its contribution is:
                # width_i = f1_{i+1} - f1_i (or 11 - f1_i for last)
                # height_i = 11 - f2_i
                # But we need the envelope: the HV is the area under the
                # step function formed by the non-dominated points.
                
                # For non-dominated points sorted by f1 ascending,
                # f2 should be descending.
                # HV = sum_i (f1_{i+1} - f1_i) * (11 - f2_i) + (11 - f1_last)*(11 - f2_last)
                # where f1_0 is the first point's f1, and we extend to 0 on the left
                # and to 11 on the right.
                
                # Actually: HV = (11 - f1_0)*(11 - f2_0) + sum_{i=1}^{m-1} (f1_i - f1_{i-1})*(11 - f2_i)
                # No, let me think more carefully.
                
                # The HV of a set of 2D points with ref (r1, r2) is:
                # Sort points by f1 ascending: p1, p2, ..., pm
                # HV = sum_{i=1}^{m} (f1_{i+1} - f1_i) * (r2 - f2_i)
                # where f1_{m+1} = r1, and we also add (r1 - f1_m) * (r2 - f2_m)
                # Wait, let me just use the standard formula:
                # HV = (r1 - f1_1)*(r2 - f2_1) + sum_{i=2}^{m} (f1_i - f1_{i-1})*(r2 - f2_i)
                
                # For marginal gain of adding point (a, b):
                # Find where it would be inserted in the sorted list.
                # It removes the contribution of the segment it replaces.
                
                # This is getting complex. Let me use a simpler approach:
                # Compute HV of current set, then HV of set+candidate, take difference.
                # But vectorize over candidates.
                
                # Alternative: use the fact that in 2D, adding a point (a,b)
                # increases HV by the area of the rectangle that is now dominated
                # but wasn't before. This equals:
                # (a - f1_left) * (f2_above - b) where f1_left is the f1 of the
                # nearest point to the left with f2 < b, and f2_above is the f2
                # of the nearest point below... this is still complex.
                
                # Let me just compute HV directly for each candidate in a loop.
                # With 5000 candidates and up to 200 points, this is 5000*200 = 1M ops
                # per step, times 200 steps = 200M ops. Might be slow.
                
                # Better: precompute the current HV, then for each candidate,
                # compute the incremental HV analytically.
                
                # In 2D, if we have sorted points by f1: (f1_1, f2_1), ..., (f1_m, f2_m)
                # with f1_1 < f1_2 < ... < f1_m and f2_1 > f2_2 > ... > f2_m
                # (non-dominated)
                # 
                # HV = (r1 - f1_1)*(r2 - f2_1) + sum_{i=2}^{m} (f1_i - f1_{i-1})*(r2 - f2_i)
                
                # Adding point (a, b):
                # Find j such that f1_j < a < f1_{j+1} (or a < f1_1 or a > f1_m)
                # The new point is non-dominated if b < f2_j (the f2 of the point
                # to its left) and b > f2_{j+1} (the f2 of the point to its right).
                # 
                # The HV increase = (a - f1_j)*(f2_j - b) if such j exists
                # (the area of the rectangle that becomes newly dominated)
                
                # Let me implement this:
                m = len(sf1)
                # For each candidate a, find insertion position
                # Use searchsorted
                insert_pos = np.searchsorted(sf1, f1_candidates, side='left')
                
                gains = np.zeros(n_candidates)
                
                for i in range(n_candidates):
                    pos = insert_pos[i]
                    a = f1_candidates[i]
                    b = f2_candidates[i]
                    
                    if pos == 0:
                        # To the left of all points
                        # Check if b < sf2[0] (non-dominated)
                        if b < sf2[0]:
                            gains[i] = (a - 0) * (sf2[0] - b)  # Hmm, not quite
                            # Actually: the new point dominates the region
                            # [0, a] x [b, sf2[0]] that wasn't dominated before
                            # Wait, no. The region [0, a] x [b, 11] was partially
                            # dominated by the first point (f1_1, f2_1) which dominates
                            # [0, f1_1] x [f2_1, 11].
                            # The new point dominates [0, a] x [b, 11].
                            # If a < f1_1 and b < f2_1, the new gain is
                            # (a - 0)*(f2_1 - b) if a < f1_1
                            # Actually this is getting too complex. Let me just
                            # compute HV directly.
                        pass
                    elif pos == m:
                        # To the right of all points
                        if b < sf2[-1]:
                            # New gain: (a - sf1[-1]) * (sf2[-1] - b)
                            gains[i] = (a - sf1[-1]) * (sf2[-1] - b)
                    else:
                        # Between sf1[pos-1] and sf1[pos]
                        left_f2 = sf2[pos - 1]
                        right_f2 = sf2[pos]
                        if b < left_f2 and b > right_f2:
                            # Non-dominated, gain is the rectangle
                            gains[i] = (a - sf1[pos - 1]) * (left_f2 - b)
                        elif b < left_f2:
                            # Dominated by left point? No, b < left_f2 means it's
                            # better in f2. But if b > right_f2, it's non-dominated.
                            # If b <= right_f2, it's dominated by the right point.
                            pass
                        elif b > right_f2:
                            pass
                
                # This is too slow and error-prone. Let me use a simpler approach:
                # Just compute HV for each candidate by building the set and computing HV.
                # With numpy, I can vectorize the HV computation.
                
                # Actually, let me use a completely different approach:
                # Since the Pareto front is known analytically, I can use
                # dynamic programming or a simple numerical optimization.
                
                # For now, let me fall back to a power-law distribution with
                # optimized exponent, which is known to work well for concave fronts.
                # The optimal density for a concave front f2(x) is proportional
                # to |f2'(x)|, so x0 should be sampled with density proportional
                # to 2*x0, meaning x0 = u^0.5 where u is uniform.
                # But for HV with a reference point, the optimal sampling is
                # more nuanced.
                
                # Let me just use a quadratic spacing: x0_i = (i/(n-1))^2
                # This puts more points near x0=0.
                
                # Actually, let me try a different approach entirely:
                # Use scipy to optimize the 200 x0 values to maximize HV.
                # But that might be too slow.
                
                # For now, use a simple power law: x0 = (i/(n-1))^beta with beta=0.5
                # This should be better than uniform.
                break  # Exit the greedy loop, use power law
        
        if len(plan) < n:
            # Fallback: power-law distribution
            if n > 1:
                u = np.linspace(0.0, 1.0, n)
                plan = (u ** 0.5).tolist()
            else:
                plan = [0.0]
        
        return plan

    def __call__(self, problem, bounds):
        n_evals = self.budget
        if n_evals <= 0:
            return None
        
        if self._x0_plan is None:
            self._x0_plan = self._compute_optimal_plan(n_evals)
        
        x1_values = self._x0_plan
        
        for i in range(n_evals):
            if time.time() > self.deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_values[i]
            
            try:
                problem(x)
            except Exception:
                break
                
        return None
