import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 300s limit
        
        # Strategy: Deterministic Global Optimization with Sobol sequence
        # This maximizes early AOCC by ensuring rapid coverage of the search space.
        
        # Generate Sobol sequence points
        # We need 'budget' points in [0, 1)^dim
        # scipy.stats.qmc.Sobol is available in newer SciPy versions.
        # If not, we can use a simple low-discrepancy sequence or random with stratification.
        # Let's try to use scipy if available, otherwise fallback to random stratified sampling.
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=False)
            # We want exactly 'budget' points.
            # Sobol requires power of 2 for best properties, but we can take first N.
            # Actually, for better uniformity, it's often good to take the first 2^k points where 2^k >= budget.
            # But we have a strict budget limit.
            # Let's generate enough points.
            n_points = 1 << (int(np.ceil(np.log2(budget))))
            if n_points < budget:
                n_points = budget
            points = sampler.random(n_points)
            # Map to [-5, 5]^dim
            points = points * 10.0 - 5.0
            # Take the first 'budget' points
            points = points[:budget]
        except ImportError:
            # Fallback: Stratified random sampling
            # Divide the space into grid cells and pick one random point per cell
            # For dim=5, budget=500, we can do e.g. 3^5 = 243 cells, or 4^5 = 1024 cells.
            # Let's do a simple random sampling but ensure diversity.
            # Actually, simple random is okay but Sobol is better for AOCC.
            # Let's use a simple Latin Hypercube or just random.
            # Given the constraints, let's stick to random if SciPy fails, but try to make it robust.
            points = np.random.uniform(-5, 5, size=(budget, dim))
        
        best_x = points[0]
        best_f = func(best_x)
        calls_made = 1
        
        for i in range(1, budget):
            if time.time() > deadline:
                break
            x = points[i]
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        return best_x, best_f
