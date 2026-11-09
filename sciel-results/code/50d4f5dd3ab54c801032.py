"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic Sobol-based global search with adaptive local refinement.
    
    Strategy:
    1. Generate a Sobol sequence of points in [0,1]^dim, map to [-5,5]^dim.
    2. Evaluate points in order of space-filling quality (Sobol is already well-spread).
    3. After the initial sweep, if budget remains, do a local refinement around the best point.
    
    Since we have exactly 500 budget and dim=5, we can use a Sobol sequence of 512 points
    (next power of 2 for efficiency) and take the first 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Use SciPy's Sobol sequence for quasi-random space-filling points
        from scipy.stats import qmc
        
        # Sobol requires power-of-2 sample sizes for best properties
        # We'll generate 512 points (next power of 2 >= 500) and use first 500
        n_sobol = 512
        sobol = qmc.Sobol(d=dim, scramble=False)
        points_unit = sobol.random(n_sobol)
        
        # Map from [0, 1] to [-5, 5]
        points = points_unit * 10.0 - 5.0
        
        # Evaluate all Sobol points
        best_x = None
        best_f = float('inf')
        
        calls_made = 0
        for i in range(min(n_sobol, budget)):
            x = points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # If we have remaining budget, do local refinement around best point
        remaining = budget - calls_made
        if remaining > 0 and best_x is not None:
            # Local search with decreasing step sizes
            # Use a few iterations of local random search around best_x
            rng = np.random.default_rng(42)
            sigma = 0.5
            for _ in range(remaining):
                # Generate perturbation
                direction = rng.standard_normal(dim)
                y = best_x + sigma * direction
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                # Keep sigma reasonable
                sigma = max(0.01, min(sigma, 2.0))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
