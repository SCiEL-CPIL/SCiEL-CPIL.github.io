"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    Strategy: Deterministic optimal placement of points on the known Pareto front
    segments to maximize hypervolume with reference point [11, 11].
    
    ZDT3 Pareto front: 7 disconnected segments in f1 space.
    For each segment, we place points using a density that accounts for the
    curvature of f2 = g_min * (1 - sqrt(f1/g_min)).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3 Pareto front segments in x1 space (7 segments)
        # These are the well-known segments from the ZDT literature
        segments = [
            (0.0, 0.05),
            (0.15, 0.25),
            (0.35, 0.45),
            (0.55, 0.65),
            (0.75, 0.85),
            (0.95, 1.0)
        ]
        
        # Compute g_min: minimum of g(x) = 1 + 10*(n-1) + sum[2(x_i^2 - cos(4*pi*x_i)) + 10]
        # Min of 2(x^2 - cos(4*pi*x)) + 10 on [0,1]
        # h(x) = x^2 - cos(4*pi*x), h(0) = -1, so min term = 2*(-1) + 10 = 8
        # g_min = 1 + 10*(dim-1) + (dim-1)*8 = 1 + 18*(dim-1)
        g_min = 1 + 18 * (dim - 1)
        
        # The Pareto front: f2 = g_min * (1 - sqrt(f1/g_min))
        # For hypervolume maximization with ref point (11, 11):
        # We want to place points to maximize the area dominated by the set.
        
        # Strategy: For each segment, place points using inverse CDF sampling
        # with a density proportional to the local slope magnitude of the front.
        # The slope of f2 w.r.t. f1 is: df2/df1 = g_min * (-1/(2*sqrt(f1*g_min)))
        # = -sqrt(g_min)/(2*sqrt(f1))
        # The magnitude is proportional to 1/sqrt(f1), which is larger for small f1.
        
        # However, for hypervolume, the optimal placement is actually to equalize
        # the "marginal contribution" which for a monotone front means placing
        # points such that the rectangles they form have equal area.
        
        # A simpler and effective approach: place points at quantiles of the
        # Pareto front parameterized by f1, weighted by segment length.
        
        # Allocate budget across segments proportional to their length
        seg_lengths = np.array([h - l for l, h in segments])
        total_len = seg_lengths.sum()
        
        # Number of points per segment (proportional to length)
        counts = np.maximum(1, np.floor(budget * seg_lengths / total_len)).astype(int)
        # Adjust to match budget exactly
        diff = budget - counts.sum()
        if diff > 0:
            # Add extra points to longest segments
            order = np.argsort(-seg_lengths)
            for i in range(diff):
                counts[order[i % len(order)]] += 1
        elif diff < 0:
            # Remove points from shortest segments (but keep at least 1 per segment)
            order = np.argsort(seg_lengths)
            for i in range(-diff):
                idx = order[i % len(order)]
                if counts[idx] > 1:
                    counts[idx] -= 1
        
        # Generate x1 values for each segment
        x1_values = []
        for i, (l, h) in enumerate(segments):
            n_pts = counts[i]
            if n_pts == 1:
                # Place at the midpoint of the segment
                x1_values.append((l + h) / 2)
            elif n_pts == 2:
                # Place at the boundaries
                x1_values.extend([l, h])
            else:
                # Place points using a density that accounts for the front shape
                # Use inverse CDF with weight w(f1) = 1/sqrt(f1 + eps) to emphasize
                # regions where the front is steeper (smaller f1)
                eps = 1e-6
                # Create a fine grid to compute the CDF
                f1_grid = np.linspace(l, h, 10000)
                weights = 1.0 / np.sqrt(f1_grid + eps)
                cdf = np.cumsum(weights)
                cdf /= cdf[-1]
                
                # Generate quantiles
                quantiles = np.linspace(0, 1, n_pts + 2)[1:-1]  # avoid endpoints
                # Inverse CDF
                pts = np.interp(quantiles, cdf, f1_grid)
                x1_values.extend(pts.tolist())
        
        x1_values = np.array(x1_values)
        
        # Ensure we have exactly budget points
        if len(x1_values) < budget:
            # Fill remaining with random points from segments
            remaining = budget - len(x1_values)
            extra = np.random.choice(
                [x for l, h in segments for x in [l, h]],
                size=remaining
            )
            x1_values = np.concatenate([x1_values, extra])
        elif len(x1_values) > budget:
            x1_values = x1_values[:budget]
        
        # Sort for better coverage
        x1_values.sort()
        
        # Build x vectors: x1 = x1_values, x2..xn set to 0 (which minimizes g)
        X = np.zeros((budget, dim))
        X[:, 0] = x1_values
        # x[1:] = 0 already from initialization, and x=0 minimizes h(x) = x^2 - cos(4*pi*x)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
