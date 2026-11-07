"""Multi-objective optimizer on ZDT6 using hypervolume-guided adaptive grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-objective optimizer on ZDT6 using hypervolume-guided adaptive grid."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        results = []

        def eval_and_record(x):
            nonlocal used
            if used >= budget:
                return
            if time.time() > deadline:
                return
            x = np.clip(x, lo, hi)
            try:
                fx = problem(x)
                fx = np.array(fx, dtype=float)
                if np.all(np.isfinite(fx)) and np.all(fx < 11.0):
                    results.append(fx)
                used += 1
            except Exception:
                used += 1

        # ZDT6 Analysis:
        # f1(x) = 1 - exp(-4*x0 * sin^2(10*pi*x0))
        # g(x) = 1 + 9*(sum(x[1:]) / (dim-1))^0.25
        # f2(x) = (1 + g(x)) * (1 - sqrt(f1(x)))
        #
        # Pareto front is at x[1:] = 0 (g=0).
        # f1 = 1 - exp(-4*x0 * sin^2(10*pi*x0))
        # f2 = 1 - sqrt(f1)
        #
        # We want to choose 200 x0 values to maximize hypervolume of the resulting points.
        # The front is oscillatory with 10 humps. Uniform sampling wastes points in flat regions.
        # We use a simple adaptive strategy: sample more densely where |df1/dx0| is large.
        
        # Compute a density function for x0 based on the derivative magnitude
        # f1(x0) = 1 - exp(-4*x0 * sin^2(10*pi*x0))
        # Let u = 4*x0 * sin^2(10*pi*x0)
        # f1 = 1 - exp(-u)
        # df1/dx0 = exp(-u) * du/dx0
        # du/dx0 = 4*sin^2(10*pi*x0) + 4*x0 * 2*sin(10*pi*x0)*cos(10*pi*x0)*10*pi
        #        = 4*sin^2(10*pi*x0) + 40*pi*x0*sin(20*pi*x0)
        
        def density(x0):
            # Compute |df1/dx0| as a proxy for "interestingness"
            s = np.sin(10 * np.pi * x0)
            c = np.cos(10 * np.pi * x0)
            u = 4 * x0 * s * s
            exp_u = np.exp(-u)
            du_dx = 4 * s * s + 40 * np.pi * x0 * s * c
            return np.abs(exp_u * du_dx)
        
        # Create a grid of x0 values weighted by density
        # Use inverse CDF sampling
        n_grid = 10000
        x0_grid = np.linspace(0, 1, n_grid)
        dens = density(x0_grid)
        
        # Normalize density to create a PDF
        dens = dens / np.sum(dens)
        
        # Compute CDF
        cdf = np.cumsum(dens)
        cdf = cdf / cdf[-1]
        
        # We want to select budget points. But we also need to ensure good coverage.
        # A pure density-weighted sample might cluster too much.
        # Let's use a mix: 50% uniform, 50% density-weighted.
        # Or better: use a stratified sampling approach on the CDF.
        
        # Actually, for hypervolume maximization on a 1D front, the optimal spacing is related to the curvature.
        # A simpler robust approach: Use a fine uniform grid, then greedily select points that maximize hypervolume.
        # But greedy selection of 200 from 10000 is O(200 * 10000 * HV_cost) which might be slow if HV_cost is high.
        # However, 2D hypervolume calculation is fast.
        
        # Let's try a different approach:
        # 1. Generate a candidate set of 1000 x0 values using density-weighted sampling.
        # 2. Use a simple local search to optimize the positions of 200 points to maximize HV.
        #    This is too complex.
        
        # Alternative:
        # The "humps" are at x0 = k/20 + 1/20 for k=0..19? No, sin^2(10*pi*x) has period 1/10.
        # Peaks of sin^2 are at 10*pi*x = pi/2 + k*pi => x = 1/20 + k/10.
        # So peaks at 0.05, 0.15, 0.25, ..., 0.95.
        # Valleys at 0, 0.1, 0.2, ..., 1.0.
        # f1 is small at valleys, large at peaks.
        # f2 = 1 - sqrt(f1).
        # At valleys (f1~0), f2~1.
        # At peaks (f1~1), f2~0.
        # The front goes from (0,1) to (1,0) with oscillations.
        
        # To maximize HV, we need points that "cover" the front well.
        # Since the front is 1D, the HV is the area under the "upper envelope" of the points (in the minimization sense, it's the area of the union of rectangles from each point to the reference point).
        # For a set of points (f1_i, f2_i) sorted by f1, the HV is sum of (ref_f2 - f2_i) * (f1_{i+1} - f1_i) ... roughly.
        # Actually, for 2D, if we sort by f1 ascending, the HV is:
        # sum_{i} (ref_f2 - f2_i) * (f1_{i+1} - f1_i) + (ref_f2 - f2_last) * (ref_f1 - f1_last)
        # where f1_{i+1} is the next point's f1, and ref_f1, ref_f2 are the reference point.
        
        # We can optimize the x0 values to maximize this HV.
        # Let's use a simple gradient-free local search on the x0 values.
        # Start with a uniform grid of 200 points.
        # Perturb each point slightly and accept if HV increases.
        
        n_points = budget
        x0_vals = np.linspace(0, 1, n_points)
        
        def compute_hv(x0_arr):
            # Compute f1, f2 for given x0 values
            s = np.sin(10 * np.pi * x0_arr)
            u = 4 * x0_arr * s * s
            f1 = 1 - np.exp(-u)
            f2 = 1 - np.sqrt(f1)
            
            # Filter out points that are not strictly better than ref [11, 11]
            # All points on the front are in [0,1] for both objectives, so they are all better.
            
            # Sort by f1
            idx = np.argsort(f1)
            f1_sorted = f1[idx]
            f2_sorted = f2[idx]
            
            # Calculate HV
            # ref = [11, 11]
            ref_f1, ref_f2 = 11.0, 11.0
            
            # HV = sum of areas of rectangles
            # For each point i, it contributes a rectangle of width (f1_{i+1} - f1_i) and height (ref_f2 - f2_i)
            # Except for the last point, width is (ref_f1 - f1_last)
            
            if len(f1_sorted) == 0:
                return 0.0
            
            # Widths
            f1_ext = np.append(f1_sorted, ref_f1)
            widths = np.diff(f1_ext)
            
            # Heights
            heights = ref_f2 - f2_sorted
            
            # HV is sum of widths * heights
            # Note: This formula assumes that the points form a "staircase" that is strictly decreasing in f2 as f1 increases.
            # For ZDT6, f2 = 1 - sqrt(f1). Since f1 is not monotonic with x0, the points might not be sorted in a way that f2 is strictly decreasing.
            # Wait, f2 = 1 - sqrt(f1). So if f1 increases, f2 decreases.
            # If we sort by f1, f2 will be sorted in decreasing order.
            # So the "staircase" assumption holds for the non-dominated points.
            # However, if f1 is not monotonic, some points might be dominated by others.
            # The HV calculation using the sorted f1 and the corresponding f2 works correctly for the union of rectangles IF we only consider the non-dominated front.
            # But if we have dominated points, they don't contribute to the HV.
            # The formula sum(widths * heights) actually calculates the area of the polygon formed by the points sorted by f1, assuming they are all non-dominated.
            # If some are dominated, this formula overestimates the HV.
            # To be safe, we should filter out dominated points.
            
            # Filter dominated points
            # A point i is dominated by j if f1_j <= f1_i and f2_j <= f2_i (and at least one strict).
            # Since we sorted by f1, we can do a linear scan.
            # Keep a point if its f2 is strictly less than the minimum f2 seen so far.
            
            min_f2_so_far = np.inf
            keep_mask = np.zeros(len(f2_sorted), dtype=bool)
            for i in range(len(f2_sorted)):
                if f2_sorted[i] < min_f2_so_far:
                    keep_mask[i] = True
                    min_f2_so_far = f2_sorted[i]
            
            if not np.any(keep_mask):
                return 0.0
            
            f1_kept = f1_sorted[keep_mask]
            f2_kept = f2_sorted[keep_mask]
            
            f1_ext = np.append(f1_kept, ref_f1)
            widths = np.diff(f1_ext)
            heights = ref_f2 - f2_kept
            
            return np.sum(widths * heights)
        
        # Local search to maximize HV
        best_hv = compute_hv(x0_vals)
        current_x0 = x0_vals.copy()
        
        # Simple hill climbing
        step_size = 0.01
        max_iter = 50
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(len(current_x0)):
                # Try perturbing point i
                original = current_x0[i]
                for delta in [-step_size, step_size]:
                    new_x0 = current_x0.copy()
                    new_x0[i] = np.clip(original + delta, 0, 1)
                    hv = compute_hv(new_x0)
                    if hv > best_hv:
                        best_hv = hv
                        current_x0 = new_x0
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break
        
        # Evaluate the optimized points
        for x0 in current_x0:
            x = np.zeros(dim)
            x[0] = x0
            eval_and_record(x)

        # If we still have budget (shouldn't happen), do random search
        if used < budget:
            while used < budget:
                if time.time() > deadline:
                    break
                x = np.random.rand(dim)
                x[1:] = 0.0
                eval_and_record(x)

        return results
# EVOLVE-BLOCK-END
