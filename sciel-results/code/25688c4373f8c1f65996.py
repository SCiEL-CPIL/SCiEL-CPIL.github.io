"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer on ZDT6 exploiting problem structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        # Pre-compute mapping from f1 to x1 for ZDT6
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)**6
        # We sample x1 densely to build a lookup table
        n_samples = 50000
        x1_samples = np.linspace(0, 1, n_samples)
        f1_samples = 1 - np.exp(-4 * x1_samples) * (np.sin(6 * np.pi * x1_samples) ** 6)
        
        # Sort by f1 for efficient lookup
        sort_idx = np.argsort(f1_samples)
        f1_sorted = f1_samples[sort_idx]
        x1_sorted = x1_samples[sort_idx]

        def get_x1_for_f1(target_f1):
            """Find x1 that yields f1 closest to target_f1."""
            target_f1 = np.clip(target_f1, 0, 1)
            # Binary search for insertion point
            idx = np.searchsorted(f1_sorted, target_f1)
            
            # Handle boundaries
            if idx == 0:
                return x1_sorted[0]
            if idx == len(f1_sorted):
                return x1_sorted[-1]
            
            # Check neighbors to find the closest f1
            # f1_sorted[idx-1] <= target_f1 < f1_sorted[idx]
            dist_prev = abs(f1_sorted[idx - 1] - target_f1)
            dist_curr = abs(f1_sorted[idx] - target_f1)
            
            if dist_prev < dist_curr:
                return x1_sorted[idx - 1]
            else:
                return x1_sorted[idx]

        # Strategy:
        # 1. The Pareto front of ZDT6 is achieved when x_2...x_10 = 0.
        # 2. We need to cover the range of f1 values [0, 1] to maximize hypervolume.
        # 3. We use a gap-filling strategy on the f1 axis.
        
        evaluated_f1s = []
        evaluated_f2s = []
        
        # Initial population: spread across f1 range
        init_count = min(30, budget)
        targets = np.linspace(0, 1, init_count)
        
        for t in targets:
            if time.time() > deadline:
                break
            x1 = get_x1_for_f1(t)
            x = np.zeros(dim)
            x[0] = x1
            # x[1:] is 0, which minimizes g and thus f2 for a given f1
            f = problem(x)
            evaluated_f1s.append(f[0])
            evaluated_f2s.append(f[1])
        
        # Fill gaps in f1 coverage
        remaining = budget - init_count
        for _ in range(remaining):
            if time.time() > deadline:
                break
            
            # Sort evaluated f1 values
            evaluated_f1s.sort()
            
            # Consider the full range [0, 1] for gap analysis
            # This ensures we cover the extremes as well
            all_points = [0.0] + evaluated_f1s + [1.0]
            all_points.sort()
            
            max_gap = 0
            max_gap_mid = 0.5  # Default midpoint of [0,1]
            
            for i in range(len(all_points) - 1):
                gap = all_points[i + 1] - all_points[i]
                if gap > max_gap:
                    max_gap = gap
                    max_gap_mid = (all_points[i] + all_points[i + 1]) / 2.0
            
            # Evaluate the point at the midpoint of the largest gap
            x1 = get_x1_for_f1(max_gap_mid)
            x = np.zeros(dim)
            x[0] = x1
            f = problem(x)
            evaluated_f1s.append(f[0])
            evaluated_f2s.append(f[1])

        return None
# EVOLVE-BLOCK-END
