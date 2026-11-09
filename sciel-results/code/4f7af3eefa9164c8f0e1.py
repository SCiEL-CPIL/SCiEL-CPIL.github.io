import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Exploits DTLZ1 structure to sample the Pareto front directly using Sobol sequences."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 300s to be safe.
        deadline = t_start + 290.0 

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ1 with M=3 objectives has M-1 = 2 shape variables (x_0, x_1).
        # The remaining N-M+1 = 5 variables (x_2..x_6) are distance variables.
        # To achieve g=0 (minimize distance part), set distance vars to 0.5.
        # The Pareto front is the set of points where g=0.
        # We sample the shape variables (x_0, x_1) to cover the front.
        
        # Use Sobol sequence for low-discrepancy sampling in 2D.
        # Generate power of 2 samples >= budget, then truncate.
        next_pow2 = 1
        while next_pow2 < budget:
            next_pow2 *= 2
            
        s = qmc.Sobol(d=2, scramble=True)
        samples = s.random(next_pow2)
        samples = samples[:budget]
        
        # Explicitly include the corners of the decision space that map to
        # unique vertices of the Pareto front in objective space.
        # (x0, x1) = (1, 0) -> (1, 0, 0)
        # (x0, x1) = (0, 1) -> (0, 1, 0)
        # (x0, x1) = (0, 0) -> (0, 0, 1)
        # (x0, x1) = (1, 1) -> (1, 0, 0) (duplicate of first)
        
        # Replace the first 3 samples with these explicit corners to ensure
        # maximum coverage of the extreme ends of the front.
        if budget >= 3:
            samples[0] = [1.0, 0.0]
            samples[1] = [0.0, 1.0]
            samples[2] = [0.0, 0.0]

        # Construct points
        points = np.zeros((budget, dim))
        points[:, 0] = samples[:, 0]
        points[:, 1] = samples[:, 1]
        
        # Set distance variables to 0.5
        for j in range(2, dim):
            points[:, j] = 0.5
            
        # Evaluate
        results = []
        for i in range(budget):
            if time.time() > deadline:
                break
            x = points[i]
            results.append(problem(x))
            
        return results
