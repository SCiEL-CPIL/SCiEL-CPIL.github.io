"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Multi-objective optimizer using Deterministic Hierarchical Sobol Sampling.
    Designed for WFG4 (12 dims, 3 objectives) with a budget of 200 evaluations.
    
    Strategy:
    - Use a Sobol sequence to generate a low-discrepancy set of points.
    - This ensures systematic coverage of the 12-dimensional space, which is 
      often superior to random or evolutionary methods for fixed budgets in 
      high dimensions, as it avoids clustering and gaps.
    - The first N points of a Sobol sequence are a nested design, meaning 
      adding more points refines the coverage rather than just adding noise.
    - We simply evaluate the first 'budget' points from the Sobol sequence.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate Sobol sequence
        # scipy.stats.qmc.Sobol requires dim >= 1
        # We want 'budget' points.
        # Sobol sequence generation is fast.
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        
        # Generate slightly more points than budget to ensure we have enough 
        # valid points if any were to be filtered (though Sobol points are always valid 
        # within [0,1]^d).
        # Note: scipy's Sobol generator produces points in [0, 1).
        # We need 'budget' points.
        n_points = self.budget
        
        # Generate the points
        # scramble=False is deterministic. 
        # However, the task says "np.random.seed(seed)" is called. 
        # Sobol with scramble=False ignores np.random state.
        # If we want to respect the seed for reproducibility across seeds 0 and 1, 
        # we should probably use scramble=True or rely on the fact that 
        # the evaluator seeds np.random. 
        # But Sobol sequences are deterministic by nature. 
        # Using scramble=False means seed 0 and seed 1 will evaluate the EXACT same points.
        # This is a valid strategy: deterministic sampling.
        # The score is the mean over two seeds. If both are identical, the mean is that score.
        
        # Let's use scramble=False for pure determinism and low discrepancy.
        # The Sobol generator in SciPy uses a specific algorithm (Joe & Kuo) which is fixed.
        
        try:
            # n_points must be a power of 2 for the generator to work efficiently? 
            # No, scipy handles arbitrary n, but it might generate up to the next power of 2 
            # and slice. 
            # Actually, scipy.stats.qmc.Sobol.random(n) generates n points.
            unit_points = sampler.random(n_points)
        except Exception:
            # Fallback to random if Sobol fails for some reason
            unit_points = np.random.rand(n_points, self.dim)

        # Map from [0, 1)^d to bounds
        # x = lo + (hi - lo) * u
        range_ = hi - lo
        X = lo + range_ * unit_points
        
        # Evaluate all points
        # We must be careful not to exceed the budget.
        # We generated exactly 'budget' points.
        
        # Time check
        if time.time() > deadline:
            return None
            
        F = np.array([problem(x) for x in X])
        
        # The return value is ignored, but we return something valid
        return X

# EVOLVE-BLOCK-END
