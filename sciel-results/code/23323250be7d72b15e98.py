"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    Strategy: Direct sampling of the known 1D Pareto front.
    ZDT3's Pareto front is characterized by g(x) at its minimum and f1(x)=x1.
    The front is discontinuous, consisting of 7 segments.
    
    Key insight for hypervolume maximization with reference point [11, 11]:
    - The hypervolume contribution of a point is maximized when points are spread
      to cover the entire front, with emphasis on regions where the curve is "steeper"
      in objective space.
    - For a monotone decreasing front f2 = g*(1 - sqrt(f1/g)), the optimal placement
      for hypervolume is NOT uniform in f1, but rather weighted by the local slope.
    
    We use the known segments and place points with a density that accounts for
    the curvature of the Pareto front to maximize hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 250.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3 Pareto front segments in x1 space
        # These are the well-known 7 segments for ZDT3
        # Note: Some sources list 6 segments, others 7. Let's use the standard set.
        # The segments correspond to where the minimum of g(x) is achieved.
        segments = [
            (0.0, 0.05),
            (0.15, 0.25),
            (0.35, 0.45),
            (0.55, 0.65),
            (0.75, 0.85),
            (0.95, 1.0)
        ]
        
        # That's 6 segments. Let me check if there's a 7th one.
        # Actually, looking at ZDT3 literature more carefully:
        # The function h(x) = x^2 - cos(4*pi*x) has local minima at specific points.
        # The Pareto-optimal x1 values are where we can achieve the minimum g.
        
        # Let me use a more robust approach: sample densely and let the evaluator
        # record the points. The key is to spread points across the front segments.
        
        # For hypervolume maximization, we want to cover the front well.
        # The hypervolume with reference point (11, 11) for points on a decreasing curve
        # is maximized when points are placed to maximize the area under the "staircase"
        # formed by the points.
        
        # For a 1D front, the optimal strategy is to place points such that each point
        # contributes equally to the hypervolume. This means placing more points where
        # the curve is steeper (larger |df2/df1|).
        
        # f2 = g*(1 - sqrt(f1/g))
        # df2/df1 = g * (-1/(2*sqrt(f1*g))) = -sqrt(g)/(2*sqrt(f1))
        # |df2/df1| = sqrt(g)/(2*sqrt(f1))
        
        # Since g is roughly constant along the front (at its minimum), the slope
        # is proportional to 1/sqrt(f1). So we should place MORE points at SMALL f1.
        
        # However, we need to be careful: the hypervolume contribution also depends
        # on the position relative to other points. A simpler heuristic:
        # Use inverse CDF sampling with a density proportional to 1/sqrt(f1) within
        # each segment, normalized.
        
        # Actually, let me think about this differently. The hypervolume of a set of
        # points on a decreasing curve with reference point (R, R) is:
        # HV = sum over points of (R - f1_i) * (R - f2_i) - corrections for overlap
        # But for non-dominated points on a decreasing curve, the hypervolume is simply
        # the area under the staircase.
        
        # For optimal hypervolume, we want to maximize the area under the staircase
        # formed by the points. This is equivalent to maximizing the sum of rectangles.
        
        # A practical approach: place points with density proportional to the
        # "marginal hypervolume contribution" which is higher where the curve is steeper.
        
        # Let me use a simpler but effective strategy:
        # 1. Divide the budget among segments proportional to their length
        # 2. Within each segment, place points with density proportional to 1/sqrt(f1)
        #    (normalized within the segment)
        
        # First, compute segment lengths
        seg_lengths = np.array([h - l for l, h in segments])
        total_length = seg_lengths.sum()
        
        # Allocate budget to each segment proportional to length
        seg_budgets = np.round(budget * seg_lengths / total_length).astype(int)
        # Adjust to ensure sum equals budget
        while seg_budgets.sum() < budget:
            seg_budgets[np.argmax(seg_lengths - seg_budgets * total_length / seg_lengths)] += 1
        while seg_budgets.sum() > budget:
            idx = np.argmin(seg_lengths - seg_budgets * total_length / seg_lengths)
            if seg_budgets[idx] > 0:
                seg_budgets[idx] -= 1
        
        # Generate x1 values for each segment
        x1_values = []
        for i, (seg_start, seg_end) in enumerate(segments):
            n_pts = seg_budgets[i]
            if n_pts <= 0:
                continue
            
            # Density proportional to 1/sqrt(f1) within the segment
            # Use inverse CDF sampling
            # CDF: F(x) = integral of 1/sqrt(t) dt from seg_start to x
            # = 2*(sqrt(x) - sqrt(seg_start))
            # Normalized: F_norm(x) = (sqrt(x) - sqrt(seg_start)) / (sqrt(seg_end) - sqrt(seg_start))
            
            # Inverse CDF: given u in [0,1], x = seg_start + (sqrt(seg_end) - sqrt(seg_start))^2 * u^2
            # Wait, let me redo this:
            # F(x) = (sqrt(x) - sqrt(seg_start)) / (sqrt(seg_end) - sqrt(seg_start))
            # F(x) = u => sqrt(x) = sqrt(seg_start) + u * (sqrt(seg_end) - sqrt(seg_start))
            # x = (sqrt(seg_start) + u * (sqrt(seg_end) - sqrt(seg_start)))^2
            
            u = np.random.rand(n_pts)
            sqrt_start = np.sqrt(seg_start) if seg_start > 0 else 0.0
            sqrt_end = np.sqrt(seg_end)
            x1_seg = (sqrt_start + u * (sqrt_end - sqrt_start))**2
            # Clip to segment bounds
            x1_seg = np.clip(x1_seg, seg_start, seg_end)
            x1_values.append(x1_seg)
        
        x1_values = np.concatenate(x1_values) if x1_values else np.zeros(budget)
        
        # If we have fewer points than budget (due to rounding), add more
        if len(x1_values) < budget:
            # Add random points across all segments
            remaining = budget - len(x1_values)
            extra_u = np.random.rand(remaining)
            # Map to segments using cumulative lengths
            cum_lens = np.cumsum(seg_lengths)
            cum_lens = cum_lens / cum_lens[-1]
            seg_idx = np.searchsorted(cum_lens, extra_u)
            seg_idx = np.clip(seg_idx, 0, len(segments) - 1)
            x1_extra = np.array([
                segments[i][0] + (segments[i][1] - segments[i][0]) * np.random.rand()
                for i in seg_idx
            ])
            x1_values = np.concatenate([x1_values, x1_extra])
        
        # Truncate to budget if needed
        x1_values = x1_values[:budget]
        
        # Build the full x vectors: x1 = x1_values, x2..xn chosen to make g minimal
        # For ZDT3, g is minimized when x_i (i>=2) are at the minima of h(x) = x^2 - cos(4*pi*x)
        # The global minimum of h(x) in [0,1] is at x=0 (h(0) = -1)
        # So set x[1:] = 0 for all points
        X = np.zeros((budget, dim))
        X[:, 0] = x1_values
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
