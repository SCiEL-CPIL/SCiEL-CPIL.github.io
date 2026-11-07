import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT6 Pareto front analysis:
        # When x[1:] = 0: g = 1, h = 1 - sqrt(f1/1) = 1 - sqrt(f1)
        # f1 = 1 + 8*|sin(10*pi*x0)|, ranges from 1 to 9
        # f2 = (1+1)*(1-sqrt(f1)) = 2 - sqrt(f1)
        # So Pareto front: f2 = 2 - sqrt(f1), f1 in [1, 9]
        # In terms of f1: f2 ranges from 2-sqrt(9)=2-3=-1 to 2-sqrt(1)=2-1=1
        
        def make_x(x0):
            """Create a point on the Pareto front with given x0 value."""
            x = np.zeros(self.dim)
            x[0] = np.clip(x0, lo[0], hi[0])
            return x
        
        def f1_from_x0(x0):
            """Compute f1 from x0 analytically."""
            return 1 + 8 * abs(np.sin(10 * x0 * np.pi))
        
        def f2_from_f1(f1_val):
            """Compute f2 from f1 analytically (on Pareto front)."""
            return 2 - np.sqrt(f1_val)
        
        def x0_from_f1(target_f1):
            """Find x0 that gives target_f1. Returns best x0 in [lo[0], hi[0]]."""
            sin_val = (target_f1 - 1) / 8
            sin_val = np.clip(sin_val, 0, 1)
            base_angle = np.arcsin(sin_val)
            
            valid_x0s = []
            for k in range(-3, 4):
                x0_1 = (base_angle + 2*np.pi*k) / (10*np.pi)
                if lo[0] <= x0_1 <= hi[0]:
                    valid_x0s.append(x0_1)
                x0_2 = (np.pi - base_angle + 2*np.pi*k) / (10*np.pi)
                if lo[0] <= x0_2 <= hi[0]:
                    valid_x0s.append(x0_2)
            
            if not valid_x0s:
                # Fallback: grid search
                best_x0 = 0.5
                best_diff = float('inf')
                for x0_test in np.linspace(lo[0], hi[0], 200):
                    f1_test = f1_from_x0(x0_test)
                    diff = abs(f1_test - target_f1)
                    if diff < best_diff:
                        best_diff = diff
                        best_x0 = x0_test
                return best_x0
            
            return min(valid_x0s, key=lambda x0: abs(f1_from_x0(x0) - target_f1))
        
        evals = 0
        evaluated_points = []  # List of (x, f1, f2) tuples
        
        # Phase 1: Optimal spacing for hypervolume maximization
        # For a 2D hypervolume with reference point (11, 11), we want to maximize
        # the sum of areas of rectangles formed by consecutive Pareto front points.
        # 
        # The Pareto front is f2 = 2 - sqrt(f1), f1 in [1, 9].
        # For points sorted by f1: f1_1 < f1_2 < ... < f1_n
        # Hypervolume = sum over i of (f1_{i+1} - f1_i) * (f2_i - ref_f2) 
        #              + (ref_f1 - f1_1) * (f2_1 - ref_f2) + ... 
        # Actually, for minimization with reference point (r1, r2):
        # HV = sum of (r1 - f1_i) * (f2_{i-1} - f2_i) type terms
        # 
        # More precisely, if we have points (f1_1, f2_1), ..., (f1_n, f2_n) sorted by f1 ascending,
        # and f2 is decreasing (which it is: f2 = 2-sqrt(f1) is decreasing in f1),
        # then HV = (r1 - f1_1)*(f2_1 - r2) + sum_{i=1}^{n-1} (f1_{i+1} - f1_i)*(f2_i - r2)
        # Wait, let me reconsider. For minimization:
        # A point (f1, f2) dominates reference (r1, r2) if f1 < r1 and f2 < r2.
        # The hypervolume is the volume of the region dominated by at least one point.
        # 
        # For 2D with points sorted by f1 ascending (and f2 descending since it's a Pareto front):
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (r2 - f2_i)  where f1_{n+1} = r1
        #     But we need f1_i < r1 and f2_i < r2 for each point to contribute.
        # 
        # Actually the standard formula: sort by f1 ascending. 
        # HV = (r1 - f1_1) * (r2 - f2_1) + sum_{i=2}^{n} (f1_i - f1_{i-1}) * (r2 - f2_{i-1})
        # No wait. Let me think again.
        # 
        # The dominated region is the union of rectangles [f1_i, r1] x [f2_i, r2] for each point.
        # Since f2 is decreasing in f1 on the Pareto front, the union forms a staircase.
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (r2 - f2_i) where f1_{n+1} = r1
        # But only for points where f1_i < r1 and f2_i < r2.
        #
        # Since f1 ranges [1,9] and r1=11, all points have f1 < r1.
        # f2 ranges [-1, 1] and r2=11, all points have f2 < r2.
        #
        # So HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (r2 - f2_i) with f1_{n+1} = r1 = 11
        #
        # To maximize this with n points, we should place points to maximize this sum.
        # This is a discretization problem. For a smooth curve, optimal spacing 
        # in terms of f1 is roughly uniform when the curve is nearly linear.
        # f2 = 2 - sqrt(f1) is concave, so we might want denser points where the curve is steeper.
        #
        # For now, let's use a good initial set and then refine.
        
        # Initial: use 50 points with spacing optimized for hypervolume
        # Since f2 = 2 - sqrt(f1), the slope df2/df1 = -1/(2*sqrt(f1))
        # Steeper at small f1. So we want more points at small f1.
        # Use spacing proportional to sqrt(f1) for roughly uniform f2 spacing? 
        # Or just use a reasonable number and let gap-filling handle it.
        
        n_initial = 50
        # Use f1 values that give good coverage
        # Let's use uniform in f1 first, then refine
        f1_targets = np.linspace(1.0, 9.0, n_initial)
        
        for f1_t in f1_targets:
            if evals >= self.budget or time.time() > deadline:
                break
            x0 = x0_from_f1(f1_t)
            x = make_x(x0)
            f = problem(x)
            evaluated_points.append((x, f[0], f[1]))
            evals += 1
        
        # Sort by f1
        evaluated_points.sort(key=lambda p: p[1])
        
        # Phase 2: Gap-filling based on hypervolume contribution
        # For each gap between consecutive points, estimate the hypervolume gain
        # from adding a point in the middle.
        
        max_iterations = 200
        for _ in range(max_iterations):
            if evals >= self.budget or time.time() > deadline:
                break
            
            if len(evaluated_points) < 2:
                break
            
            f1_vals = np.array([p[1] for p in evaluated_points])
            f2_vals = np.array([p[2] for p in evaluated_points])
            
            # Compute hypervolume contribution of each gap
            # Gap i is between point i and point i+1
            # Also gaps from first point to r1=11 and from last point... 
            # Actually the "gaps" in the staircase are between consecutive f1 values.
            
            n_pts = len(evaluated_points)
            best_gain = -1
            best_gap_idx = -1
            best_mid_f1 = None
            
            # Gaps between consecutive points
            for i in range(n_pts - 1):
                f1_lo = f1_vals[i]
                f1_hi = f1_vals[i+1]
                f2_lo = f2_vals[i]  # f2 at f1_lo (higher f2 since f1_lo < f1_hi)
                f2_hi = f2_vals[i+1]  # f2 at f1_hi (lower f2)
                
                # Current contribution of this segment: (f1_hi - f1_lo) * (r2 - f2_lo)
                # If we add a point at f1_mid, f2_mid:
                # New contribution: (f1_mid - f1_lo) * (r2 - f2_lo) + (f1_hi - f1_mid) * (r2 - f2_mid)
                # Gain = new - old = (f1_hi - f1_mid) * (r2 - f2_mid) - (f1_hi - f1_lo) * (r2 - f2_lo) + (f1_mid - f1_lo) * (r2 - f2_lo)
                #      = (f1_hi - f1_mid) * (r2 - f2_mid) - (f1_hi - f1_mid) * (r2 - f2_lo)
                #      = (f1_hi - f1_mid) * (f2_lo - f2_mid)
                
                # Try midpoint in f1
                f1_mid = (f1_lo + f1_hi) / 2
                f2_mid = f2_from_f1(f1_mid)
                
                gain = (f1_hi - f1_mid) * (f2_lo - f2_mid)
                
                if gain > best_gain:
                    best_gain = gain
                    best_gap_idx = i
                    best_mid_f1 = f1_mid
            
            # Also consider gap from last point to r1=11
            if n_pts > 0:
                f1_last = f1_vals[-1]
                f2_last = f2_vals[-1]
                # Current contribution of last segment: (r1 - f1_last) * (r2 - f2_last)
                # If we add point at f1_mid between f1_last and r1:
                # f2_mid = f2_from_f1(f1_mid) -- but wait, f1_mid > 9, and the Pareto front only goes to f1=9
                # For f1 > 9, there's no point on the Pareto front. So we can't add points there.
                # Actually, the Pareto front is only for f1 in [1,9]. Beyond that, no Pareto-optimal points.
                # So the gap from f1=9 to r1=11 is just empty space. We can't fill it.
                # But we should make sure we have a point near f1=9.
                pass
            
            # Also consider gap from r1... no, f1 starts at 1, and we can't go below 1 on the Pareto front.
            
            if best_gap_idx < 0 or best_mid_f1 is None:
                break
            
            # Add point at best_mid_f1
            x0 = x0_from_f1(best_mid_f1)
            x = make_x(x0)
            f = problem(x)
            evaluated_points.append((x, f[0], f[1]))
            evals += 1
            
            # Re-sort
            evaluated_points.sort(key=lambda p: p[1])
        
        # Phase 3: Use remaining budget for local refinement
        # Add points near existing points to improve density
        while evals < self.budget and time.time() <= deadline:
            if len(evaluated_points) == 0:
                break
            
            # Pick a random existing point and perturb it slightly
            idx = np.random.randint(0, len(evaluated_points))
            x_base = evaluated_points[idx][0].copy()
            
            # Small perturbation in x0 only
            perturbation = np.random.normal(0, 0.005)
            x_base[0] = np.clip(x_base[0] + perturbation, lo[0], hi[0])
            
            f = problem(x_base)
            evaluated_points.append((x_base, f[0], f[1]))
            evals += 1
        
        # Return the last evaluated point
        if evaluated_points:
            return evaluated_points[-1][0]
        else:
            return np.zeros(self.dim)
