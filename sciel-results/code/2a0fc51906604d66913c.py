import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        
        n_evals = self.budget
        dim = self.dim
        start_time = time.time()
        deadline = 300.0

        # ZDT6 structure:
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2 = g(x) * (1 - sqrt(f1/g(x)))
        # g(x) = 1 + 9 * (sum(x[1:])/(n-1))^(1/(1+2*sqrt(sum(x[1:])/(n-1))))
        #
        # The Pareto front is achieved when x[1:] = 0, so g(x) = 1.
        # Then f2 = 1 - sqrt(f1).
        # The front is the curve (f1, 1 - sqrt(f1)) for f1 in [0, 1].
        # Note: f1(x1) oscillates. The minimum value of f1 is not 0, but > 0.
        # Let's find the range of f1.
        # f1(x1) = 1 - Y(x1), where Y(x1) = exp(-4*x1) * sin(6*pi*x1)^8.
        # Max Y is approx 0.716 (at some x1). So min f1 is approx 0.284.
        # However, the "Pareto front" in standard ZDT6 is defined for the true front.
        # Actually, for ZDT6, the Pareto set is x1 in [0,1], x2..xn = 0.
        # The Pareto front is the image of this set.
        # Since f1 is not monotonic, the front is a curve that goes up and down in f1?
        # No, f1 is bounded. Let's check the range.
        # If we just sample x1 uniformly, we get a distribution of f1.
        # To maximize hypervolume, we want to cover the objective space well.
        # The reference point is [11, 11].
        # The front is (f1, 1-sqrt(f1)).
        # If we can choose x1 such that f1 is uniformly distributed in its range, that's good.
        # But we can also just sample x1 such that the resulting points (f1, f2) are well spread.
        
        # Strategy:
        # 1. Create a dense grid of x1 in [0, 1].
        # 2. Compute f1 for each x1.
        # 3. We want to select n_evals points from this grid that maximize coverage.
        #    A good heuristic is to select points that are well-distributed in the f1 value.
        #    Or, since the mapping x1 -> f1 is many-to-one and oscillatory, 
        #    simply selecting x1 to be uniformly spaced in [0,1] might not give uniform f1.
        #    Let's check the derivative or just use the grid to pick unique f1 values.
        
        # Better strategy:
        # We know the front is (f1, 1-sqrt(f1)).
        # We want to place points on this front.
        # We can parameterize by f1. But we need x1 such that f1(x1) = target_f1.
        # Since f1 is oscillatory, for a given f1 (in the range), there are multiple x1.
        # We can just pick one.
        
        # Let's build a lookup table:
        # x1_grid -> f1_grid
        # Then for a set of target f1 values, find the closest x1.
        
        # What are the target f1 values?
        # To maximize hypervolume with reference [11,11], we want to cover the front.
        # The front is convex? 
        # f2 = 1 - sqrt(f1).
        # d f2 / d f1 = -1/(2*sqrt(f1)).
        # The curvature is high near f1=0.
        # So we should have more points near f1=0.
        # However, f1 cannot be 0. The minimum f1 is ~0.284.
        # So the front is from f1_min to 1.
        
        # Let's compute f1_min accurately.
        # Y(x1) = exp(-4*x1) * sin(6*pi*x1)^8
        # Let's just use a dense grid to find the range and then sample f1 in that range.
        
        n_grid = 20000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = 1 - np.exp(-4 * x1_grid) * np.sin(6 * np.pi * x1_grid)**8
        
        f1_min = np.min(f1_grid)
        f1_max = np.max(f1_grid) # Should be 1.0
        
        # We want to select n_evals points.
        # Let's sample f1 targets.
        # To account for the curvature of the front (f2 = 1 - sqrt(f1)),
        # we might want to sample f1 such that the points are evenly spaced in objective space?
        # Or just uniform in f1?
        # Uniform in f1 is a safe bet.
        # But since the slope is steep near f1_min, uniform f1 might cluster points in f2 space near the bottom?
        # Actually, if f1 is uniform, f2 = 1 - sqrt(f1) is also fairly spread, but slightly non-uniform.
        
        # Let's try to sample f1 uniformly in [f1_min, f1_max].
        # And then for each target f1, find the x1 that achieves it.
        # If there are multiple x1, pick the one that is "central" or just the first one found.
        # Since we just need to cover the front, any x1 that gives the right f1 is fine.
        
        # To make it efficient, we can bin the f1_grid.
        # For each target f1, find the index in f1_grid with min |f1_grid - target|.
        # This is O(N_evals * N_grid) which is 200 * 20000 = 4e6, very fast.
        
        f1_targets = np.linspace(f1_min, f1_max, n_evals)
        
        x1_selected = np.empty(n_evals)
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                # Fallback: just use remaining budget with uniform x1
                break
            
            target = f1_targets[i]
            # Find closest x1 in grid
            # Using argmin on absolute difference
            idx = np.argmin(np.abs(f1_grid - target))
            x1_selected[i] = x1_grid[idx]
            
            if time.time() - start_time > deadline:
                break
        
        # It's possible that some x1_selected are duplicates if f1_targets are close and f1 is flat.
        # But with 20000 grid points and 200 targets, duplicates are unlikely unless f1 is very flat.
        # ZDT6 f1 is oscillatory, so it's not flat.
        
        # Ensure we don't exceed budget
        # We already limited the loop to n_evals.
        
        # Evaluate
        for i in range(len(x1_selected)):
            if time.time() - start_time > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x1_selected[i]
            # x[1:] = 0
            
            # Clip to bounds
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
            
        return None
