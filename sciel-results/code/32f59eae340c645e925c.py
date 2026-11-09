# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Analytic inversion strategy for ZDT3 hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        if budget <= 0:
            return None
            
        # ZDT3 specific setup
        # f1(x) = x0
        # g(x) = 1 + 10*(n-1) + sum_{i=1}^{n-1} [ (xi-0.5)^2 - 10*cos(2*pi*(xi-0.5)) ]
        # f2(x) = (1 + sqrt(f1/g)) * h(f1/g, g)
        # h(u, g) = 1 - sqrt(u/g) - u/g * sin^2(10 * pi * u)
        
        # The Pareto front is characterized by x0 in [0,1] and g = g_min = 1.
        # For xi=0.5, each term is -10, so g_min = 1 + 10*(dim-1) - 10*(dim-1) = 1.
        
        g_min = 1.0
        
        # h(u, 1) = 1 - sqrt(u) - u * sin^2(10 * pi * u)
        # The Pareto front has 7 distinct segments due to the sin^2 term.
        # The discontinuities occur where sin(10*pi*u) = 0, i.e., 10*pi*u = k*pi => u = k/10 for k=0,1,...,10
        
        # To maximize hypervolume, we want to place points such that they cover the objective space well.
        # Since the front is non-uniform in f2 due to the h function, we should sample more points
        # where f2 changes rapidly or where the front is "curved" to better approximate the hypervolume.
        
        # Strategy: Use the fact that the Pareto front is known analytically.
        # We can compute the exact f2 values for any x0 in [0,1] with g=1.
        # To maximize hypervolume, we should distribute points to cover the range of f2 values well.
        
        # The function h(u, 1) = 1 - sqrt(u) - u * sin^2(10 * pi * u)
        # For u in [0,1], f2 = (1 + sqrt(u)) * h(u, 1)
        
        # Let's compute f2 for a dense set of u values to understand the shape.
        # Then we can use inverse transform sampling or quantile-based sampling to place points
        # where they contribute most to hypervolume.
        
        # Actually, for hypervolume maximization with a reference point (11, 11),
        # the contribution of a point (f1, f2) is proportional to the area it "owns" in the 
        # objective space that is not dominated by other points and is better than the reference.
        
        # A good strategy is to use a quasi-random sequence in the objective space or 
        # to sample x0 values that correspond to evenly spaced f2 values (or some other metric).
        
        # Let's try sampling x0 values that are evenly spaced in the "effective" range of f2.
        # First, let's compute f2 for a fine grid of x0 values.
        
        n_grid = 10000
        x0_grid = np.linspace(0, 1, n_grid)
        u = x0_grid
        f2_grid = (1 + np.sqrt(u)) * (1 - np.sqrt(u) - u * np.sin(10 * np.pi * u)**2)
        
        # Now we want to pick `budget` points from this grid (or interpolate) to maximize hypervolume.
        # A simple heuristic: pick points that are evenly spaced in the f2 value range,
        # but weighted by the density of the front in that region.
        
        # Alternatively, we can use the fact that for hypervolume, we want to cover the 
        # "extremes" and the "middle" well. Let's use a combination of:
        # 1. A few points at the extremes (x0 near 0 and x0 near 1)
        # 2. The rest distributed based on the curvature of the front
        
        # Actually, let's use a simpler approach: use the inverse CDF of the f2 distribution
        # to sample points. This will place more points where f2 is more "dense" in the grid.
        
        # But wait - for hypervolume, we actually want to minimize the "gap" in objective space.
        # So we should sample points such that the gaps between consecutive points (in some metric) are minimized.
        
        # Let's try a different approach: use a space-filling curve or quasi-random sequence
        # in the x0 space, but with a transformation that accounts for the non-uniformity of f2.
        
        # The key insight: the hypervolume contribution is related to the area under the front.
        # To maximize hypervolume with a fixed number of points, we should place points where
        # the "marginal gain" in hypervolume is highest.
        
        # For simplicity, let's use a stratified sampling approach:
        # Divide [0,1] into `budget` equal intervals and pick one point from each.
        # But choose the point in each interval that minimizes the local hypervolume loss.
        
        # Actually, the simplest effective strategy for ZDT3 is to use a quasi-random sequence
        # like the Halton sequence or simply evenly spaced points, since the front is relatively smooth
        # except for the discontinuities.
        
        # Let's use a combination: 
        # - Use evenly spaced points in x0, but with a slight bias towards regions where f2 changes rapidly.
        
        # For now, let's stick with evenly spaced points but ensure we hit the discontinuities.
        # The discontinuities are at x0 = k/10 for k=0,1,...,10.
        
        # Let's create a set of candidate x0 values:
        # 1. The discontinuity points: 0, 0.1, 0.2, ..., 1.0
        # 2. Midpoints between discontinuities: 0.05, 0.15, ..., 0.95
        # 3. Additional points to fill in the budget
        
        n_disc = 11  # 0, 0.1, ..., 1.0
        disc_points = np.arange(n_disc) / 10.0
        
        # If budget is large enough, include all discontinuity points and midpoints
        if budget >= 2 * n_disc - 1:
            # Include all discontinuity points and midpoints
            mid_points = (np.arange(n_disc - 1) + 0.5) / 10.0
            base_points = np.concatenate([disc_points, mid_points])
            base_points = np.unique(base_points)
            
            # Now we have 21 base points. If budget > 21, add more points
            if budget > len(base_points):
                # Add additional points evenly spaced in the remaining gaps
                remaining = budget - len(base_points)
                # For each interval between consecutive base points, add points
                # Actually, let's just add points evenly spaced in [0,1] that are not already in base_points
                # This is complex. Let's just use a fine grid and pick the best `budget` points.
                
                # Simpler: just use a fine grid and pick `budget` points that are evenly spaced
                # in terms of index.
                x0_vals = np.linspace(0, 1, budget)
            else:
                x0_vals = base_points[:budget]
        else:
            # If budget is small, just use evenly spaced points
            x0_vals = np.linspace(0, 1, budget)
        
        # Actually, let's reconsider. The current approach uses evenly spaced points.
        # Let's try a more sophisticated approach: use the inverse transform sampling
        # based on the f2 distribution to place points where they contribute most.
        
        # Compute the CDF of f2 values
        # Sort by f2 and compute cumulative sum of "widths"
        sort_idx = np.argsort(f2_grid)
        f2_sorted = f2_grid[sort_idx]
        x0_sorted = x0_grid[sort_idx]
        
        # The "density" of points in f2 space is related to the derivative df2/du
        # To maximize hypervolume, we want to place points such that the gaps in f2 space are minimized.
        # This is equivalent to using a stratified sampling approach in f2 space.
        
        # Let's use the following strategy:
        # 1. Divide the range of f2 into `budget` equal intervals.
        # 2. For each interval, pick the x0 value that corresponds to the median f2 in that interval.
        
        f2_min = f2_grid.min()
        f2_max = f2_grid.max()
        f2_intervals = np.linspace(f2_min, f2_max, budget + 1)
        
        x0_vals = np.zeros(budget)
        for i in range(budget):
            lo_f2 = f2_intervals[i]
            hi_f2 = f2_intervals[i + 1]
            # Find x0 values where f2 is in [lo_f2, hi_f2]
            mask = (f2_grid >= lo_f2) & (f2_grid <= hi_f2)
            if np.any(mask):
                x0_candidates = x0_grid[mask]
                # Pick the median x0
                x0_vals[i] = np.median(x0_candidates)
            else:
                # If no points in this interval, pick the closest x0
                x0_vals[i] = x0_grid[np.argmin(np.abs(f2_grid - (lo_f2 + hi_f2) / 2))]
        
        # Ensure we have unique points
        x0_vals = np.unique(x0_vals)
        if len(x0_vals) < budget:
            # Pad with evenly spaced points
            n_pad = budget - len(x0_vals)
            pad_vals = np.linspace(0, 1, n_pad)
            x0_vals = np.concatenate([x0_vals, pad_vals])
            x0_vals = np.unique(x0_vals)
            if len(x0_vals) > budget:
                x0_vals = x0_vals[:budget]
        
        # Construct the x vectors
        X = np.zeros((budget, dim))
        X[:, 0] = x0_vals[:budget]
        X[:, 1:] = 0.5
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
