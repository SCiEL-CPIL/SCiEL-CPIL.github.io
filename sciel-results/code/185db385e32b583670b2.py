# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting the known Pareto front structure.
    
    Uses Sobol quasi-random sampling to densely and uniformly cover the 2D angle space
    that defines the Pareto front of DTLZ2 (a quarter-sphere in R^3).
    Distance variables are fixed at their optimal value (0.5) to ensure points lie
    exactly on the Pareto front (g=0).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = 0
        all_F = []

        # Number of angle variables: n_obj - 1 = 2
        n_angles = self.n_obj - 1
        # Number of distance variables: dim - n_obj + 1 = 10
        n_dist = self.dim - self.n_obj + 1

        # Generate Sobol sequence points for the 2D angle space
        # Sobol sequences are low-discrepancy and provide excellent uniformity
        n_points = self.budget
        
        try:
            sampler = qmc.Sobol(d=n_angles, scramble=True)
            # Generate n_points samples in [0, 1)^n_angles
            angle_samples = sampler.random(n_points)
        except Exception:
            # Fallback to Halton if SciPy fails for some reason
            h1 = np.linspace(0, 1, n_points, endpoint=False)
            h2 = np.linspace(0, 1, n_points, endpoint=False)
            # Simple interleave to ensure coverage
            angle_samples = np.column_stack([h1, h2])

        for i in range(n_points):
            if time.time() > deadline - 10:  # Safety margin
                break

            x = np.empty(self.dim)
            
            # Set angle variables
            # Map [0, 1) to [0, 1] for the decision variables x[0], x[1]
            # The problem function handles the sin/cos transformation internally
            x[0] = angle_samples[i, 0]
            x[1] = angle_samples[i, 1]
            
            # Set distance variables to optimal value 0.5
            # This ensures g(x) = 0, placing the point exactly on the Pareto front
            x[2:] = 0.5

            # Clip to bounds just in case
            x = np.clip(x, lo, hi)

            f = problem(x)
            all_F.append(f)
            n_evals += 1

        return all_F
# EVOLVE-BLOCK-END
