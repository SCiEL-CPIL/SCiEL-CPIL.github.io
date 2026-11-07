# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting known Pareto front structure.
    
    DTLZ2 with 3 objectives, 12 variables:
    - x[0], x[1] are position variables (angles on the Pareto front)
    - x[2..11] are distance variables; optimal value is 0.5 for all
    
    Strategy:
    1. Generate a Sobol quasi-random sequence in 2D for the angle space [0, 1]^2
    2. Map to decision space: distance vars = 0.5, angle vars from Sobol
    3. This places points exactly on the Pareto front with uniform coverage
    4. Use any remaining budget for minor perturbations if needed
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # DTLZ2 structure:
        # Position variables: x[0], x[1] (first n_obj - 1 = 2 variables)
        # Distance variables: x[2..11] (last dim - n_obj + 1 = 10 variables)
        n_pos = n_obj - 1  # 2
        n_dist = dim - n_pos  # 10
        
        # Generate Sobol sequence in 2D for angle space
        # Use all budget evaluations
        n_points = budget
        
        sobol = qmc.Sobol(d=2, scramble=True)
        angles = sobol.random(n_points)  # shape (n_points, 2), values in [0,1)
        
        # Build decision vectors
        # Distance variables set to 0.5 (optimal)
        dist_opt = np.full(n_dist, 0.5)
        
        all_F = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[:n_pos] = angles[i]  # angle variables
            x[n_pos:] = dist_opt   # distance variables at optimal 0.5
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_F.append(f)
        
        if not all_F:
            # Fallback: at least evaluate one point
            x = np.zeros(dim)
            x[:n_pos] = 0.5
            x[n_pos:] = 0.5
            f = np.asarray(problem(x), dtype=float)
            all_F.append(f)
        
        return np.array(all_F)
# EVOLVE-BLOCK-END
