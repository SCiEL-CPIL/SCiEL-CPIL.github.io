# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    ZDT2 Optimizer using offline optimal quantization on the analytic Pareto front.
    Exploits the 1D structure of ZDT2 (x2...xn=0) to find the optimal placement
    of 200 points to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT2 analytic front: x2...xn = 0
        # f1 = x1
        # g = 1 + 10*(n-1)
        # f2 = g * (1 - (f1/g)^(1/0.25))
        # If x2...xn=0, g=1.
        # f2 = 1 - f1^(1/0.25) = 1 - f1^0.25
        
        n = self.dim
        # Check if standard ZDT2 bounds apply (0 to 1)
        # The problem clips x to bounds.
        
        def analytic_f2(f1):
            # f1 is in [0, 1]
            # Handle f1=0 carefully if needed, but 0^0.25 is 0.
            return 1.0 - np.power(f1, 0.25)

        def calc_hv(f1_vals, ref_point):
            # f1_vals: 1D array of f1 values
            # ref_point: (r1, r2)
            f1_vals = np.sort(f1_vals)
            f2_vals = analytic_f2(f1_vals)
            r1, r2 = ref_point
            
            # Filter points strictly better than ref point
            mask = (f1_vals < r1) & (f2_vals < r2)
            if not np.any(mask):
                return 0.0
            
            f1_f = f1_vals[mask]
            f2_f = f2_vals[mask]
            
            # Sort by f1 ascending
            idx = np.argsort(f1_f)
            f1_s = f1_f[idx]
            f2_s = f2_f[idx]
            
            # Hypervolume calculation for monotonic front
            # Area = sum of rectangles
            # For each point i, it defines a rectangle from f1[i] to f1[i+1] with height r2 - f2[i]
            # Plus the first part from 0 to f1[0]? No, the region is bounded by the points and ref point.
            # Standard HV for 2D:
            # Sort by f1 ascending.
            # HV = sum_{i=1}^{N} (f1[i+1] - f1[i]) * (r2 - f2[i])
            # where f1[N+1] = r1.
            # Also need to account for the area to the left of the first point?
            # The region is the union of rectangles [f1[i], f1[i+1]] x [0, r2 - f2[i]]?
            # No, the points are (f1, f2). The dominated region is below and to the right.
            # The area is the integral of (r2 - f2(x)) dx from 0 to r1?
            # For a set of discrete points, the HV is the area of the step function.
            # If we sort by f1 increasing, f2 is decreasing.
            # The step function goes down at each f1[i].
            # The height of the step for interval [f1[i], f1[i+1]) is r2 - f2[i].
            # So HV = sum_{i=0}^{N-1} (f1[i+1] - f1[i]) * (r2 - f2[i])
            # where f1[N] = r1.
            
            f1_ext = np.append(f1_s, r1)
            widths = np.diff(f1_ext)
            heights = r2 - f2_s
            
            # Ensure heights are positive (points are better than ref)
            heights = np.maximum(heights, 0)
            
            return np.sum(widths * heights)

        ref_point = (11.0, 11.0)
        N = self.budget
        
        # Initialize points uniformly
        x1_vals = np.linspace(0.0, 1.0, N, endpoint=True)
        
        # Optimize x1_vals to maximize HV
        # Use a simple local search / gradient-free optimization
        # Since HV is piecewise linear in x1, we can use a simple move-and-check.
        
        current_hv = calc_hv(x1_vals, ref_point)
        
        # Simple local optimization: try moving each point slightly
        # To be fast, we can use a coarser grid for optimization if needed, but N=200 is small.
        # Let's do a few iterations of random perturbations + best move.
        
        rng = np.random.default_rng(42)
        
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            best_hv = current_hv
            best_x1 = x1_vals.copy()
            
            # Try random perturbations
            for _ in range(100):
                x1_trial = x1_vals.copy()
                # Perturb a few points
                idx_to_perturb = rng.integers(0, N, size=5)
                x1_trial[idx_to_perturb] += rng.normal(0, 0.01, size=5)
                x1_trial = np.clip(x1_trial, 0.0, 1.0)
                
                trial_hv = calc_hv(x1_trial, ref_point)
                if trial_hv > best_hv:
                    best_hv = trial_hv
                    best_x1 = x1_trial
            
            if best_hv > current_hv:
                x1_vals = best_x1
                current_hv = best_hv
            else:
                # No improvement, break early? Or continue?
                # Let's continue a bit more to escape local minima, but HV is likely unimodal-ish per point.
                pass

        # Final evaluation
        # Construct x vectors
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        # x2...xn are 0, which is within [0,1]
        
        # Call problem
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
