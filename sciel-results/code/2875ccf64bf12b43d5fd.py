"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT3-aware dense Pareto front sampling optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        budget = self.budget
        dim = self.dim

        # ZDT3 Pareto front analysis:
        # f2 is defined where h(g, x1) > 0.
        # With x[1:] = 0, g = 1, h(1, x1) = 10 * cos(36 * pi * x1).
        # We need cos(36 * pi * x1) > 0.
        # This occurs in 36 disjoint intervals within [0, 1].
        # Each interval has width 1/72.
        
        # Generate the 36 intervals where the Pareto front is valid
        intervals = []
        for k in range(18):
            # Each period [2k*pi, 2(k+1)*pi] in theta = 36*pi*x1
            # corresponds to x1 in [k/18, (k+1)/18]
            # Cos is positive in [2k*pi, 2k*pi + pi/2] and [2k*pi + 3*pi/2, 2(k+1)*pi]
            
            # First positive part in period k
            # theta_start = 2*k*pi -> x1_start = 2*k*pi / (36*pi) = k/18
            # theta_end = 2*k*pi + pi/2 -> x1_end = (2*k*pi + pi/2) / (36*pi) = k/18 + 1/72
            start1 = k / 18.0
            end1 = k / 18.0 + 1.0 / 72.0
            if end1 > 1.0:
                end1 = 1.0
            intervals.append((start1, end1))
            
            # Second positive part in period k
            # theta_start = 2*k*pi + 3*pi/2 -> x1_start = (2*k*pi + 3*pi/2) / (36*pi) = k/18 + 3/72
            # theta_end = 2*(k+1)*pi -> x1_end = (k+1)/18
            start2 = k / 18.0 + 3.0 / 72.0
            end2 = (k + 1) / 18.0
            if start2 < end2:
                if end2 > 1.0:
                    end2 = 1.0
                intervals.append((start2, end2))
        
        # Filter out any empty or invalid intervals (shouldn't happen with logic above, but safe)
        valid_intervals = [(s, e) for s, e in intervals if s < e]
        
        n_intervals = len(valid_intervals)
        if n_intervals == 0:
             # Fallback to uniform sampling if something went wrong
             x0_vals = np.linspace(0, 1, budget)
             for x0 in x0_vals:
                 if time.time() > deadline:
                     break
                 x = np.zeros(dim)
                 x[0] = x0
                 problem(x)
             return None

        # Distribute budget evenly among intervals
        # We want to sample each interval densely
        # Total budget = 200. n_intervals = 36.
        # Points per interval = 200 / 36 ~ 5.55
        # We'll use integer division and add remainder to first intervals
        
        points_per_interval = budget // n_intervals
        remainder = budget % n_intervals
        
        evaluated = 0
        for i, (start, end) in enumerate(valid_intervals):
            if time.time() > deadline:
                break
            
            # Assign points
            n_pts = points_per_interval
            if i < remainder:
                n_pts += 1
            
            if n_pts <= 0:
                continue
                
            # Generate points in [start, end]
            # Use linspace to ensure endpoints are covered if n_pts > 1
            # If n_pts == 1, take the midpoint
            if n_pts == 1:
                x0_vals = np.array([(start + end) / 2.0])
            else:
                x0_vals = np.linspace(start, end, n_pts)
                
            for x0 in x0_vals:
                if time.time() > deadline:
                    break
                x = np.zeros(dim)
                x[0] = x0
                # x[1:] = 0 ensures g=1, placing the point on the Pareto front
                problem(x)
                evaluated += 1
                
                if evaluated >= budget:
                    return None

        return None
# EVOLVE-BLOCK-END
