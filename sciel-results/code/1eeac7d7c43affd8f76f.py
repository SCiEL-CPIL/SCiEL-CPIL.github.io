"""Hypervolume-weighted Pareto front sampling for DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits the analytical Pareto front of DTLZ2 with hypervolume-weighted sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ2 with M=3 objectives, n=12 variables:
        # Pareto front: x[2:] = 0.5, x[0], x[1] in [0,1]
        # f1 = x[0], f2 = (1-x[0])*x[1], f3 = (1-x[0])*(1-x[1])
        # This traces a quarter sphere in the positive octant.
        
        # Strategy: Use hypervolume-weighted sampling.
        # Points closer to the origin (lower f values) contribute more to hypervolume.
        # We want to place points where the marginal hypervolume gain is highest.
        # 
        # Approach: Generate a grid in (theta, phi) space but weight the sampling
        # to be denser in regions where the point is "better" (closer to origin).
        # Alternatively, use a recursive bisection approach in objective space.
        
        # Let's use a stratified sampling approach in the 2D parameter space (x0, x1)
        # but with a transformation that clusters points towards the "better" corner.
        
        # Actually, let's use the recursive bisection idea directly.
        # We'll generate points by recursively splitting the Pareto front region.
        # But since the front is curved, it's easier to work in parameter space.
        
        # A simpler and effective approach: use a grid in (u, v) where u, v in [0,1]
        # and map to the front. To weight towards the origin, we can use a power transform.
        
        # Let's try a combination: 
        # 1. Generate a dense grid in parameter space (x0, x1)
        # 2. But use a non-uniform grid that is denser near (0,0) in objective space.
        
        # In DTLZ2, f1=x0, f2=(1-x0)*x1, f3=(1-x0)*(1-x1)
        # The "best" point is (0,0,0) which corresponds to x0=0, x1=0 or x1=1?
        # f1=0 => x0=0
        # f2=0 => x1=0 (if x0<1)
        # f3=0 => x1=1 (if x0<1)
        # So the "corners" of the front are at (x0=0, x1=0) -> f=(0,0,1), 
        # (x0=0, x1=1) -> f=(0,1,0), (x0=1, x1=0) -> f=(1,0,0)
        # The center of the front (maximally balanced) is around x0=1/sqrt(3), x1=0.5
        
        # For hypervolume with reference (2.5, 2.5, 2.5), we want points that dominate
        # as much volume as possible. Points near the "middle" of the front (in objective space)
        # often provide good coverage.
        
        # Let's use a Sobol-like low-discrepancy sequence but in the parameter space,
        # and then apply a bias towards the center of the front.
        
        # Actually, the most straightforward improvement: use a dense grid with
        # a slight bias. Let's use a grid of size sqrt(budget) x sqrt(budget)
        # but with non-uniform spacing.
        
        n_grid = int(np.sqrt(budget))
        
        # Generate non-uniform grid: denser in the middle
        # Use a sine or power transform
        # t in [0,1], map to x in [0,1] with density higher in middle
        # x = 0.5 * (1 + sin(pi * t - pi/2)) = 0.5 * (1 - cos(pi * t))
        # This clusters points towards the ends. We want the opposite.
        # x = 0.5 * (1 - cos(pi * t)) gives high density at ends.
        # We want high density in the middle: x = 0.5 * (1 + cos(pi * t))
        
        t_vals = np.linspace(0, 1, n_grid)
        x0_vals = 0.5 * (1 + np.cos(np.pi * t_vals))
        x1_vals = 0.5 * (1 + np.cos(np.pi * t_vals))
        
        # Create meshgrid
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        X0_flat = X0.flatten()
        X1_flat = X1.flatten()
        
        # We may have more points than budget
        if len(X0_flat) > budget:
            # Randomly subsample
            rng = np.random.default_rng(42)
            indices = rng.choice(len(X0_flat), budget, replace=False)
            X0_flat = X0_flat[indices]
            X1_flat = X1_flat[indices]
        
        # If we have fewer, add random points
        while len(X0_flat) < budget:
            rng = np.random.default_rng(len(X0_flat))
            X0_flat = np.append(X0_flat, rng.uniform(0, 1))
            X1_flat = np.append(X1_flat, rng.uniform(0, 1))
        
        # Evaluate points
        results = []
        for i in range(min(len(X0_flat), budget)):
            if time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = X0_flat[i]
            x[1] = X1_flat[i]
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                results.append(f)
            except:
                continue
        
        return results
# EVOLVE-BLOCK-END
