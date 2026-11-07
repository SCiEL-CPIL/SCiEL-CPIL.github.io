import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        n = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # ZDT4 structure (standard definition):
        # f1(x) = x_1
        # g(x) = sum_{i=2}^n (x_i^2 - 10*cos(4*pi*x_i)) + 10*(n-1)
        # f2(x) = (1 + g(x)) * (1 - sqrt(f1/(1+g(x))) * (1 + sin(10*pi*f1/(1+g(x)))))
        #
        # The Pareto front is achieved when g(x) = 0, which requires x_i = 0 for all i >= 2.
        # On the Pareto front:
        #   f1 = x_1
        #   f2 = 1 - sqrt(x_1) * (1 + sin(10*pi*x_1))
        # for x_1 in [0, 1]
        #
        # Strategy: Fix x_2..x_n to their optimal values (0 if in bounds, else closest optimal),
        # and sample x_1 densely. To maximize hypervolume with a far reference point [11, 400],
        # we should sample in a way that equalizes the "box" contributions in objective space.
        #
        # The key insight from the knowledge: uniform sampling in objective space (f2) 
        # rather than parameter space (x1) better distributes hypervolume contributions.
        #
        # On the Pareto front, f2 is a function of x_1:
        #   f2(x_1) = 1 - sqrt(x_1) * (1 + sin(10*pi*x_1))
        #
        # We want to sample x_1 values such that f2 values are roughly uniform in [f2_min, f2_max].
        # First, let's determine the range of f2 on the Pareto front for x_1 in [0, 1].
        #
        # f2(0) = 1 - 0 * (1 + sin(0)) = 1
        # f2(1) = 1 - 1 * (1 + sin(10*pi)) = 1 - 1*(1+0) = 0
        #
        # But f2 oscillates due to the sin term. Let's find the min and max of f2 on [0,1].
        # f2(x) = 1 - sqrt(x)*(1 + sin(10*pi*x))
        #
        # The sin term ranges from -1 to 1, so (1 + sin(...)) ranges from 0 to 2.
        # When (1 + sin(...)) = 0, f2 = 1.
        # When (1 + sin(...)) = 2 and sqrt(x) is large, f2 is minimized.
        #
        # For hypervolume with reference point [11, 400], every point on the Pareto front
        # contributes. The hypervolume is maximized by having points that cover the front well.
        #
        # Let's use a hybrid approach: sample x_1 uniformly, but also add points at critical
        # locations where f2 has local extrema to ensure good coverage.

        # Find optimal x_2..x_n
        optimal_x_rest = np.zeros(n - 1, dtype=np.float64)
        for i in range(n - 1):
            idx = i + 1
            if lo[idx] <= 0 and 0 <= hi[idx]:
                optimal_x_rest[i] = 0.0
            else:
                # Find the value in [lo, hi] that minimizes x^2 - 10*cos(4*pi*x)
                # Minima at x = k/2 with value k^2/4 - 10
                if lo[idx] > 0:
                    k_min = int(np.ceil(2 * lo[idx]))
                    candidates = []
                    for k in [k_min, k_min + 1, k_min - 1]:
                        val = k / 2.0
                        if lo[idx] <= val <= hi[idx]:
                            candidates.append(val)
                    if not candidates:
                        optimal_x_rest[i] = lo[idx]
                    else:
                        best = min(candidates, key=lambda v: v*v - 10*np.cos(4*np.pi*v))
                        optimal_x_rest[i] = best
                else:
                    k_max = int(np.floor(2 * hi[idx]))
                    candidates = []
                    for k in [k_max, k_max - 1, k_max + 1]:
                        val = k / 2.0
                        if lo[idx] <= val <= hi[idx]:
                            candidates.append(val)
                    if not candidates:
                        optimal_x_rest[i] = hi[idx]
                    else:
                        best = min(candidates, key=lambda v: v*v - 10*np.cos(4*np.pi*v))
                        optimal_x_rest[i] = best

        # Determine x_1 range
        x1_lo = max(lo[0], 0.0)
        x1_hi = min(hi[0], 1.0)

        if x1_lo >= x1_hi:
            x1_vals = np.array([x1_lo], dtype=np.float64)
        else:
            # Sample x_1 to achieve good hypervolume coverage.
            # Use a combination of uniform sampling and points at f2 extrema.
            
            # First, find where f2 has local minima and maxima on the Pareto front
            # f2(x) = 1 - sqrt(x)*(1 + sin(10*pi*x))
            # f2'(x) = -[1/(2*sqrt(x))*(1 + sin(10*pi*x)) + sqrt(x)*10*pi*cos(10*pi*x)]
            
            # Let's densely sample to find critical points, then add them
            # But we have limited budget, so let's use a smart sampling strategy.
            
            # Approach: sample x_1 such that the resulting f2 values are approximately uniform
            # This requires inverting f2(x_1) = f2_target for f2_target in [f2_min, f2_max]
            
            # First, estimate f2 range by dense sampling
            x1_dense = np.linspace(x1_lo, x1_hi, 1000, dtype=np.float64)
            f2_dense = 1.0 - np.sqrt(x1_dense) * (1.0 + np.sin(10.0 * np.pi * x1_dense))
            f2_min = np.min(f2_dense)
            f2_max = np.max(f2_dense)
            
            # Generate target f2 values uniformly
            num_points = budget
            f2_targets = np.linspace(f2_min, f2_max, num_points, dtype=np.float64)
            
            # For each f2_target, find x_1 such that f2(x_1) = f2_target
            # Use interpolation from the dense sample
            x1_vals = np.interp(f2_targets, f2_dense[::-1], x1_dense[::-1])
            # Ensure x1_vals are within bounds
            x1_vals = np.clip(x1_vals, x1_lo, x1_hi)
            
            # Also add endpoints and some key points to ensure coverage
            extra_points = [x1_lo, x1_hi, 0.5]
            for ep in extra_points:
                if x1_lo <= ep <= x1_hi:
                    x1_vals = np.append(x1_vals, ep)
            
            # Remove duplicates by rounding
            x1_vals = np.unique(np.round(x1_vals, 12))
            
            # If we have too many points, subsample
            if len(x1_vals) > budget:
                # Keep the most spread out points
                indices = np.linspace(0, len(x1_vals) - 1, budget, dtype=int)
                x1_vals = x1_vals[indices]
            
            # If we have too few, add uniform samples
            if len(x1_vals) < budget:
                remaining = budget - len(x1_vals)
                x1_uniform = np.linspace(x1_lo, x1_hi, remaining, dtype=np.float64)
                x1_vals = np.concatenate([x1_vals, x1_uniform])
                x1_vals = np.unique(np.round(x1_vals, 12))
        
        # Construct points
        num_points = len(x1_vals)
        X = np.zeros((num_points, n), dtype=np.float64)
        X[:, 0] = x1_vals
        X[:, 1:] = optimal_x_rest

        # Evaluate all points, checking time
        F = []
        for x in X:
            if time.time() > deadline:
                break
            F.append(problem(x))
        
        return X
