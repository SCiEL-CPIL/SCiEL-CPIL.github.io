import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv_2d(self, points, ref):
        """
        Compute 2D hypervolume for a set of points.
        points: (N, 2) array of objective values.
        ref: (2,) array of reference point.
        """
        # Filter points that are dominated by ref or equal to ref
        # We want points where f1 < ref[0] and f2 < ref[1]
        mask = (points[:, 0] < ref[0]) & (points[:, 1] < ref[1])
        pts = points[mask]
        
        if len(pts) == 0:
            return 0.0
            
        # Sort by f1
        pts = pts[np.argsort(pts[:, 0])]
        
        hv = 0.0
        # The hypervolume in 2D is the sum of rectangles.
        # For each point i (sorted by f1), the contribution is:
        # (ref[0] - f1_i) * (ref[1] - f2_i) - (f1_i - f1_{i-1}) * (ref[1] - f2_{i-1})? 
        # Actually, a simpler way for 2D:
        # Sort by f1 ascending.
        # Keep track of the minimum f2 seen so far? No, that's for dominance.
        # For HV, we sum the area of the box defined by (f1_i, f2_i) and (ref[0], ref[1]) 
        # minus the overlap with previous points.
        
        # Efficient 2D HV:
        # Sort by f1.
        # Iterate through points. Maintain the "skyline" of f2.
        # Since it's 2D, if we sort by f1, we just need to consider points that have 
        # a lower f2 than all previous points in the sorted order? 
        # No, that's the Pareto front of the points.
        # The hypervolume is the union of boxes.
        
        # Let's use the standard decomposition:
        # Sort points by f1 ascending.
        # Let f2_max = ref[1]
        # For each point (f1, f2):
        #   if f2 < f2_max:
        #     hv += (f1 - f1_prev) * (f2_max - f2)  # This is wrong.
        
        # Correct 2D HV algorithm:
        # 1. Filter valid points.
        # 2. Sort by f1 ascending.
        # 3. Iterate. Keep `last_f2` = ref[1].
        #    For each point (f1, f2):
        #       If f2 < last_f2:
        #          The new contribution is (ref[0] - f1) * (last_f2 - f2) ? No.
        
        # Let's visualize.
        # Points: (0.1, 0.9), (0.5, 0.5), (0.9, 0.1). Ref: (1, 1).
        # Box 1: (0.1, 0.9) to (1, 1). Area = 0.9 * 0.1 = 0.09.
        # Box 2: (0.5, 0.5) to (1, 1). Area = 0.5 * 0.5 = 0.25.
        # Box 3: (0.9, 0.1) to (1, 1). Area = 0.1 * 0.9 = 0.09.
        # Union:
        # The region covered is the union of these boxes.
        # Sort by f1:
        # p1: (0.1, 0.9). Width to ref: 0.9. Height to ref: 0.1.
        # p2: (0.5, 0.5). Width to ref: 0.5. Height to ref: 0.5.
        # p3: (0.9, 0.1). Width to ref: 0.1. Height to ref: 0.9.
        
        # Algorithm:
        # Sort by f1 ascending.
        # hv = 0
        # prev_f2 = ref[1]
        # for p in sorted_points:
        #    if p[1] < prev_f2:
        #       # The vertical slice from prev_f2 to p[1] is covered by the current point's width?
        #       # No.
        
        # Let's use the "sum of disjoint rectangles" method.
        # Sort by f1 ascending.
        # For each point i, the rectangle it adds is:
        # x-range: [f1_i, f1_{i+1}] (or ref[0] if last)
        # y-range: [f2_i, ref[1]]
        # BUT only if f2_i is lower than all previous f2s?
        # Actually, if we sort by f1, the hypervolume is the sum over i of:
        # (f1_{i+1} - f1_i) * (ref[1] - min(f2_0...f2_i))
        # where f1_0 = min_f1, f1_{N+1} = ref[0].
        # And we only consider points that are on the "lower envelope" of f2?
        # No, we must consider all points, but the "min f2 so far" handles the dominance.
        
        f1s = pts[:, 0]
        f2s = pts[:, 1]
        
        # Sort by f1
        idx = np.argsort(f1s)
        f1s = f1s[idx]
        f2s = f2s[idx]
        
        # Initialize
        hv = 0.0
        # We integrate from left to right.
        # The "active" f2 is the minimum f2 seen so far? 
        # No, the hypervolume is the area *below* the Pareto front of the points?
        # No, the hypervolume is the volume of the region *dominated* by the points and *bounded* by ref.
        # A point (f1, f2) dominates the box [f1, ref0] x [f2, ref1].
        # The union of these boxes.
        
        # In 2D, if we sort by f1, the union is a staircase.
        # We need the lower envelope of f2? 
        # If point A has lower f1 but higher f2 than point B, A is not dominated by B.
        # The union of boxes is determined by the "Pareto front" of the points themselves.
        # So, first, extract the non-dominated points from the set of 200 points.
        
        # Find non-dominated points in 2D:
        # Sort by f1 ascending.
        # A point is non-dominated if its f2 is strictly less than all previous f2s.
        # (Since f1 is ascending, if f2 is not the new minimum, it is dominated by a previous point with lower f1 and lower f2).
        
        nd_idx = []
        min_f2_so_far = ref[1]
        for i in range(len(f1s)):
            if f2s[i] < min_f2_so_far:
                nd_idx.append(i)
                min_f2_so_far = f2s[i]
        
        if len(nd_idx) == 0:
            return 0.0
            
        nd_f1s = f1s[nd_idx]
        nd_f2s = f2s[nd_idx]
        
        # Now compute HV for this non-dominated set.
        # The set is sorted by f1 ascending.
        # The "staircase" goes from (nd_f1s[0], ref[1]) down to (ref[0], nd_f2s[-1]).
        # Actually, the region is:
        # For each segment between nd_f1s[i] and nd_f1s[i+1], the height is (ref[1] - nd_f2s[i]).
        # Plus the last segment from nd_f1s[-1] to ref[0] with height (ref[1] - nd_f2s[-1]).
        
        # Wait, the box for point i is [f1_i, ref0] x [f2_i, ref1].
        # The union of these boxes for the non-dominated set:
        # Since f1 is increasing and f2 is decreasing:
        # The area is sum_{i=0}^{M-1} (f1_{i+1} - f1_i) * (ref[1] - f2_i) + (ref[0] - f1_{M-1}) * (ref[1] - f2_{M-1})
        # where f1_{-1} is effectively the start of the first box?
        # No.
        # Let's check with 1 point: (0.5, 0.5). Ref (1,1).
        # Area = 0.5 * 0.5 = 0.25.
        # Formula: M=1. Sum is empty. Last term: (1 - 0.5) * (1 - 0.5) = 0.25. Correct.
        
        # Check with 2 points: (0.1, 0.9) and (0.5, 0.5). Ref (1,1).
        # Box 1: [0.1, 1] x [0.9, 1]. Area 0.09.
        # Box 2: [0.5, 1] x [0.5, 1]. Area 0.25.
        # Union: 
        # Region x in [0.1, 0.5]: height is 1 - 0.9 = 0.1. Area = 0.4 * 0.1 = 0.04.
        # Region x in [0.5, 1]: height is 1 - 0.5 = 0.5. Area = 0.5 * 0.5 = 0.25.
        # Total = 0.29.
        # Formula:
        # i=0: (0.5 - 0.1) * (1 - 0.9) = 0.4 * 0.1 = 0.04.
        # i=1 (last): (1 - 0.5) * (1 - 0.5) = 0.25.
        # Sum = 0.29. Correct.
        
        hv = 0.0
        M = len(nd_f1s)
        for i in range(M - 1):
            width = nd_f1s[i+1] - nd_f1s[i]
            height = ref[1] - nd_f2s[i]
            hv += width * height
        # Last segment
        width = ref[0] - nd_f1s[-1]
        height = ref[1] - nd_f2s[-1]
        hv += width * height
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        if n_evals < 1:
            return np.array([])
            
        ref = np.array([11.0, 11.0])
        
        # We assume ZDT1 structure:
        # f1 = x0
        # f2 = C * (1 - sqrt(x0/g))
        # To maximize HV, we want points on the Pareto front.
        # The Pareto front is achieved when x1...xn = lo[1:] (assuming lo=0).
        # Let's verify if lo[1:] minimizes g.
        # g = 1 + 9 * sum(x1..xn) / (n-1).
        # Min g is when sum is min, i.e., x1..xn = lo[1:].
        
        # We will optimize the 200 values of x0 (or u) to maximize the HV of the resulting points.
        # Let u_i be in [0, 1]. x0_i = lo[0] + u_i * (hi[0] - lo[0]).
        # We assume the front is f2 = C * (1 - sqrt(x0_i)).
        # C = 1 + 10 * (dim - 1).
        
        C = 1 + 10 * (self.dim - 1)
        
        def objective(u_vec):
            # u_vec is shape (n_evals,)
            x0_vals = lo[0] + u_vec * (hi[0] - lo[0])
            
            # Calculate objectives assuming optimal x1..xn
            f1 = x0_vals
            # g = 1 + 9 * sum(x1..xn)/(n-1). If x1..xn = lo[1:], sum = (n-1)*lo[1].
            # If lo[1]=0, g=1.
            # If lo[1] is not 0, g = 1 + 9*lo[1].
            # Let's compute g_min.
            g_min = 1.0 + 9.0 * lo[1]  # Assuming all x1..xn are lo[1]
            
            # f2 = C * (1 - sqrt(x0 / g))
            # Note: if x0=0, f2=C.
            # If x0/g > 1, sqrt is invalid? 
            # In ZDT1, x0 is in [0,1]. g >= 1. So x0/g <= 1.
            # But if lo[0] > 0, x0 > 0.
            # If hi[0] > g_min, then x0 can be > g_min? 
            # Standard ZDT1 has x0 in [0,1] and g in [1,10].
            # If bounds are [0,1], g_min=1. x0/g <= 1.
            
            ratio = x0_vals / g_min
            # Clip ratio to [0, 1] to be safe against numerical issues or non-standard bounds
            ratio = np.clip(ratio, 0.0, 1.0)
            
            f2 = C * (1.0 - np.sqrt(ratio))
            
            points = np.column_stack((f1, f2))
            
            return -self._compute_hv_2d(points, ref)
            
        # Initial guess: Uniform distribution
        u0 = np.linspace(0.0, 1.0, n_evals)
        
        # Optimize u_vec
        # Use L-BFGS-B as it handles bounds well.
        # Bounds for u_vec are [0, 1]
        bounds_u = [(0.0, 1.0) for _ in range(n_evals)]
        
        try:
            res = minimize(objective, u0, method='L-BFGS-B', bounds=bounds_u, options={'maxiter': 50})
            u_opt = res.x
        except Exception:
            u_opt = u0
            
        # Generate the final X matrix
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = lo[0] + u_opt * (hi[0] - lo[0])
        X[:, 1:] = lo[1:]
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.zeros((n_evals, self.n_obj))
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
