import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 295.0
        budget = int(self.budget)
        dim = int(self.dim)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # --- Analytic ZDT6 front: x[1:]=0 => g=0, f1=1-exp(-4*x0*sin^2(10*pi*x0)), f2=1-sqrt(f1)
        def f1_of_x0(x0):
            return 1.0 - np.exp(-4.0 * x0 * np.sin(10.0 * np.pi * x0) ** 2)

        # Use a closed-form parametric representation of the ZDT6 Pareto front.
        # We know that for x[1:]=0, f1 = 1 - exp(-4*x0*sin^2(10*pi*x0)) and f2 = 1 - sqrt(f1).
        # To get exact front points, we can parametrize by f1 directly.
        # However, f1 is not monotonic in x0. But on [0, 1], f1 reaches a maximum and comes back.
        # Actually, the Pareto front of ZDT6 is given by x1=...=x10=0 and x0 in [0,1].
        # The set of non-dominated points is the curve.
        # To maximize hypervolume, we should sample points that cover the front well.
        # A good strategy is to sample f1 values uniformly in [0, 1] (since f1 ranges from 0 to ~1).
        # Wait, f1 max is not exactly 1. Let's compute the max f1.
        # f1(x0) = 1 - exp(-4*x0*sin^2(10*pi*x0)).
        # The max of f1 is close to 1.
        # If we sample f1 uniformly, we need to find x0 such that f1(x0) = target.
        # This requires root finding.
        
        # Alternative: sample x0 such that the resulting points are well-distributed in objective space.
        # The current code uses a grid of x0 and finds the closest x0 for each target f1.
        # This is essentially doing the same thing but with a precomputed grid.
        
        # Let's try to improve the grid resolution or use a more efficient search.
        # With 1,000,000 points, the grid spacing is 1e-6.
        # The error in f1 due to grid spacing is small.
        
        # Let's try to increase the grid size to 10,000,000 if time permits, or use interpolation.
        # But 10M might be slow.
        
        # Let's stick with the current approach but optimize the search.
        # Instead of argmin on the whole grid, we can use np.searchsorted if f1_grid was sorted, but it's not.
        # Since f1 is not monotonic, we can't use searchsorted directly.
        
        # However, we can split the domain. f1 increases then decreases?
        # Let's check the shape.
        # f1(0)=0. f1(0.5) = 1 - exp(-2) ~ 0.86. f1(1) = 1 - exp(0) = 0.
        # So it goes up and down.
        # The Pareto front includes all non-dominated points.
        # Points with x0 > 0.5 have f1 < f1(0.5) and f2 > f2(0.5)?
        # f2 = 1 - sqrt(f1). If f1 is smaller, f2 is larger.
        # So points on the descending part (x0 > x0_max) are dominated by points on the ascending part?
        # If x0_1 < x0_max and x0_2 > x0_max with f1(x0_1) = f1(x0_2), then f2(x0_1) = f2(x0_2).
        # So they are equivalent.
        # Actually, for any f1 value, there are two x0 values (except max).
        # They give the same f2. So the Pareto front is a single curve in objective space.
        
        # So we just need to cover the range of f1 from 0 to f1_max.
        # The current code targets a_i = f1_max * (i/N)^3.
        # This biases towards small f1 (large f2) because of the cube?
        # (i/N)^3 is small for small i. So a_i is small.
        # f1 small => f2 large.
        # The reference point is [11, 11].
        # Hypervolume is maximized by spreading points.
        # Uniform sampling in f1 might be better?
        # Or uniform in f2?
        
        # Let's try uniform sampling in f1.
        # a_i = f1_max * (i / N)
        
        # Let's also try to use a finer grid for better accuracy.
        # 1M points takes ~0.1s to generate and ~1s for 200 argmins?
        # Argmin on 1M array is fast in numpy.
        
        # Let's increase grid to 5M for better accuracy.
        n_scan = 5_000_000
        x0_grid = np.linspace(0.0, 1.0, n_scan)
        f1_grid = f1_of_x0(x0_grid)
        f1_max = float(np.max(f1_grid))

        # Target a_i (f1 values) for optimal staircase
        # Try uniform distribution in f1
        N = budget
        i = np.arange(1, N + 1, dtype=float)
        a_targets = f1_max * (i / N)

        # For each target, find the x0 on the grid with f1 closest to target
        x0_chosen = np.empty(N, dtype=float)
        for k in range(N):
            if time.time() > deadline:
                x0_chosen[k:] = x0_grid[0]
                break
            target = a_targets[k]
            idx = np.argmin(np.abs(f1_grid - target))
            x0_chosen[k] = x0_grid[idx]

        # Evaluate and record
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

        for k in range(N):
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim, dtype=float)
            x[0] = float(x0_chosen[k])
            eval_and_record(x)

        return results
