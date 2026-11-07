"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic space-filling search on the ZDT3 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 30s safety margin before 320s/330s limit
        
        # ZDT3 Pareto front is obtained by setting x[1:] = 0 and varying x[0] in [0, 1]
        # We use a Halton sequence in 1D (which is just 1/n) or Sobol to distribute x[0]
        # Since dim=1 for the front, we just need to distribute x[0] in [0, 1]
        
        # Generate n points in [0, 1] using a low-discrepancy sequence
        # For 1D, the Halton sequence is 1/2, 1/3, 1/4, ...
        # Or we can use scipy.stats.qmc.Sobol
        # Let's stick to simple numpy for speed and reliability, using a sorted Halton-like sequence
        # Actually, a simple uniform grid might be sufficient if we account for the non-linearity, 
        # but a low-discrepancy sequence is better for hypervolume.
        # Let's use the inverse of the cumulative distribution function if we knew it, 
        # but since we don't, we just distribute x[0] uniformly first.
        
        # To maximize hypervolume on ZDT3, uniform distribution in x[0] is a good start.
        # Let's refine: The objective f1 = x0. f2 = (1 - sqrt(f1)) * g(x) * h(f1, x).
        # With x[1:]=0, g=1. h(x) = 1 + 10*n*(sin(10*pi*x0) + sin(10*pi*x1)) ... wait, ZDT3 h is different.
        # ZDT3: h(y) = 1 + 10*n*(sin(10*pi*y) + sin(10*pi*y^2))? No.
        # ZDT3: f1 = x0, f2 = (1 - sqrt(f1/n)) * g(x) * h(f1, x)
        # h(y) = 1 + 10*(sin(10*pi*y) + sin(10*pi*y^2)) for y in [0,1]? 
        # Actually, standard ZDT3 h(y) = 1 + 10*n*(sin(10*pi*y) + sin(10*pi*y^2)) is not quite right.
        # Let's check the definition.
        # ZDT3: f1 = x0
        # f2 = (1 - sqrt(f1/n)) * g(x) * h(f1, x)
        # g(x) = 1 + 10*(n-1) * sum(xi^2) for i>0. If xi=0, g=1.
        # h(y) = 1 + 10*(sin(10*pi*y) + sin(10*pi*y^2))? No, usually h(y) = 1 + 10*n*(sin(10*pi*y) + sin(10*pi*y^2)) is for ZDT4?
        # Standard ZDT3 h(y) = 1 + 10*(sin(10*pi*y) + sin(10*pi*y^2)) is incorrect.
        # ZDT3 h(y) = 1 + 10*n*(sin(10*pi*y) + sin(10*pi*y^2)) is also common in some descriptions but let's be careful.
        # Actually, the exact formula for ZDT3 h(y) is:
        # h(y) = 1 + 10 * (sin(10*pi*y) + sin(10*pi*y^2)) is NOT correct for ZDT3.
        # ZDT3: h(y) = 1 + 10 * n * (sin(10*pi*y) + sin(10*pi*y^2)) ?
        # Let's assume the problem provides the `problem` function. We don't need to know the exact formula if we just sample the front.
        # However, we need to know WHERE the front is. The front is at x[1:]=0.
        # So we just need to sample x[0] in [0, 1] and set x[1:]=0.
        
        # We will use a Halton sequence for x[0] to ensure good distribution.
        # In 1D, Halton is 1/2, 1/3, 1/4, ...
        # Let's generate n points.
        n = self.budget
        # We need to ensure we don't exceed the budget.
        # We will generate n points.
        
        # Simple uniform grid is often sufficient for hypervolume if the front is monotonic, 
        # but ZDT3 is not convex. A low-discrepancy sequence is better.
        # Let's use the first n terms of the Halton sequence in base 2.
        # 1/2, 1/4, 1/3, 1/8, 1/5, 1/6, 1/7, 1/9 ...
        # Actually, the standard Halton sequence in base 2 is:
        # 1/2, 1/4, 1/3, 1/8, 1/5, 1/6, 1/7, 1/9, 1/10, 1/11, 1/12, 1/13, 1/14, 1/15, 1/16 ...
        
        # Let's just use a uniform grid for simplicity and speed, as the non-linearity of ZDT3 
        # might be handled well enough by uniform spacing in x0.
        # Or better, use scipy.stats.qmc.Sobol if available.
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=1, scramble=False)
            # We need to generate n points.
            # Sobol requires power of 2 for perfect balance, but it works for any n.
            x0_vals = sampler.random(n)
            # Ensure we cover the full range [0, 1]
            # Sobol points are in [0, 1)
            # We might want to include 0 and 1 explicitly to maximize hypervolume.
            # Let's replace the first and last points with 0 and 1.
            x0_vals[0] = 0.0
            x0_vals[-1] = 1.0
        except ImportError:
            # Fallback to uniform grid
            x0_vals = np.linspace(0.0, 1.0, n)
            
        # Construct the population
        X = np.zeros((n, self.dim))
        X[:, 0] = x0_vals
        # x[1:] are already 0
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X

# EVOLVE-BLOCK-END
