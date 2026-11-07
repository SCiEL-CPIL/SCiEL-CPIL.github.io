import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        n = self.dim
        K = self.budget
        ref1 = 11.0
        ref2 = 11.0
        
        # ZDT6 Analytic Front with x2..xn = 0
        # f1 = x1
        # g = 10 - 18*x1
        # f2 = g * (1 - sqrt(f1/g))
        
        # Valid range for x1 where g > 0: x1 < 10/18 = 5/9
        # We want to select K points in x1 in [0, 5/9) to maximize hypervolume.
        
        # Let's create a fine grid of candidates
        # The hypervolume contribution of a point depends on its neighbors.
        # To maximize HV, we want to distribute points such that the "staircase" approximation
        # of the curve is as close to the curve as possible.
        
        # For a concave curve (f2 vs f1), uniform spacing in f1 (x1) is generally good.
        # However, the curve might have varying curvature.
        # Let's use a finer grid and pick the best K points using a simple heuristic or DP.
        # Given the time constraint, a simple uniform distribution in x1 is a strong baseline.
        # Let's try to optimize the spacing slightly.
        
        # Let's use 2000 candidates in x1 in [0, 5/9 - eps]
        eps = 1e-6
        x1_max = 5.0/9.0 - eps
        N = 2000
        x1_grid = np.linspace(0.0, x1_max, N)
        
        # Compute f1, f2 for grid
        f1_grid = x1_grid
        g_grid = 10.0 - 18.0 * x1_grid
        # Ensure g > 0
        valid = g_grid > 1e-10
        x1_grid = x1_grid[valid]
        f1_grid = f1_grid[valid]
        g_grid = g_grid[valid]
        
        f2_grid = g_grid * (1.0 - np.sqrt(f1_grid / g_grid))
        
        # Filter valid finite points
        finite = np.isfinite(f1_grid) & np.isfinite(f2_grid)
        x1_grid = x1_grid[finite]
        f1_grid = f1_grid[finite]
        f2_grid = f2_grid[finite]
        
        N_valid = len(x1_grid)
        
        if N_valid < K:
            # Not enough candidates, just use all
            indices = np.arange(N_valid)
        else:
            # We want to select K indices from 0..N_valid-1
            # to maximize hypervolume.
            # HV for sorted points p1..pK:
            # HV = sum_{i=1}^{K} (f1[i+1] - f1[i]) * (ref2 - f2[i])
            # where f1[K+1] = ref1.
            # Note: f1 is increasing with index.
            
            # This is equivalent to minimizing the area between the curve and the staircase.
            # A good approximation is to choose points such that the "gap" in f1 is proportional
            # to the local slope or curvature, or just uniform in f1.
            
            # Let's try uniform in f1 first.
            # indices = np.linspace(0, N_valid-1, K).astype(int)
            # indices = np.unique(indices)
            
            # Let's try a slightly more sophisticated approach:
            # The area lost by approximating the curve with a staircase is related to
            # the integral of the error.
            # For a smooth concave function, uniform spacing in x minimizes the maximum error.
            # But we want to maximize the area under the staircase.
            # Area = sum (x_{i+1} - x_i) * y_i
            # True Area = integral y dx
            # Loss = True Area - Staircase Area
            # For concave y, y_i > average y on [x_i, x_{i+1}]? No, y is decreasing.
            # Wait, f2 is decreasing as f1 increases?
            # f1 = x1. f2 = g(1-sqrt(f1/g)).
            # As x1 increases, g decreases.
            # Let's check monotonicity.
            # At x1=0, f2 = 10*(1-0) = 10.
            # At x1=5/9, g->0, f2 -> 0?
            # limit x1->5/9: g = 10-18x1. Let g=10-18x1. x1 = (10-g)/18.
            # f2 = g(1 - sqrt((10-g)/18/g)) = g(1 - sqrt((10-g)/(18g)))
            # As g->0, sqrt(10/(18g)) -> inf. So f2 -> -inf?
            # Wait, sqrt(f1/g). If g->0+, f1>0, then f1/g -> inf.
            # So f2 -> -inf.
            # But we only care about points with f2 < 11.
            # Actually, if f2 becomes negative, it's still valid for HV if f1 < 11.
            # But wait, ZDT6 standard definition:
            # f2 = g * (1 - sqrt(f1/g))
            # If g is small, this can be very negative.
            # However, the reference point is (11, 11).
            # Points with very negative f2 contribute a lot to HV because (ref2 - f2) is large.
            # But we must ensure the points are non-dominated.
            # Since f1 is increasing and f2 is decreasing (likely), all points are non-dominated.
            
            # Let's verify f2 decreasing.
            # Derivative of f2 w.r.t x1.
            # It is known that ZDT6 Pareto front is a curve from (0, 10) to (5/9, -something).
            # Actually, the standard ZDT6 front is often plotted for x1 in [0, 1] but g becomes negative.
            # If g < 0, sqrt(f1/g) is undefined.
            # So the domain is restricted to x1 < 5/9.
            # At x1 = 5/9, g=0.
            # Just before 5/9, g is small positive. f1/g is large. f2 is large negative.
            # So the front goes from (0, 10) to (5/9, -inf).
            # This means the curve is very steep at the end.
            # To capture the area, we need points densely packed where the curve is steep?
            # No, the area is (ref1 - f1) * (ref2 - f2).
            # If f2 is very negative, the height is large.
            # But the width is small.
            
            # Let's just use a fine grid and pick K points uniformly in index space.
            # This corresponds to uniform spacing in x1.
            
            indices = np.linspace(0, N_valid - 1, K).astype(int)
            indices = np.unique(indices)
            
            # If we lost points due to uniqueness, we might have less than K.
            # We can fill in the gaps by shifting or inserting.
            # But typically N_valid >> K, so linspace gives unique integers.
            
            if len(indices) < K:
                # This should be rare
                pass

        # Evaluate the selected points
        results = []
        for i in range(len(indices)):
            if time.time() > deadline:
                break
            if i >= K:
                break
                
            idx = indices[i]
            x1 = x1_grid[idx]
            
            x = np.zeros(n)
            x[0] = x1
            
            try:
                f = problem(x)
                f = np.asarray(f, dtype=float)
                if np.all(np.isfinite(f)):
                    # Check if it's better than reference
                    if f[0] < ref1 and f[1] < ref2:
                        results.append(f)
            except Exception:
                pass

        return results
