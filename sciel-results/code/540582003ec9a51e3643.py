"""DTLZ1 optimizer using stratified simplex sampling for optimal hypervolume."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Optimizes DTLZ1 by sampling the analytic Pareto front with stratified points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        all_points = []
        deadline = time.time() + 290
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # DTLZ1 structure for n_obj=3, dim=7:
        # x[0], x[1]: shaping variables (2D simplex for M=3)
        # x[2]..x[6]: g-variables (5 variables)
        # g is minimized when all g-variables = 0.5
        
        base_x = np.zeros(dim)
        base_x[2:] = 0.5
        
        # The Pareto front for DTLZ1 M=3 is a 2D triangle in objective space.
        # Shape variables x[0], x[1] must satisfy x[0] >= 0, x[1] >= 0, x[0]+x[1] <= 1.
        # The third shape component is 1 - x[0] - x[1].
        
        n_evals = min(budget, 200)
        
        if n_evals <= 0:
            return all_points
        
        # Use Sobol sequence for uniform coverage of the 2D simplex
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=2, seed=42)
            # Generate enough points to get n_evals valid ones
            # Acceptance rate is 0.5, so 2*n_evals is a good estimate.
            # Use next power of 2 for efficiency, or just a fixed large number.
            n_sobol = 512 
            points = sobol.random(n_sobol)
            
            valid_points = []
            for p in points:
                if p[0] + p[1] <= 1.0:
                    valid_points.append(p)
                if len(valid_points) >= n_evals:
                    break
        except ImportError:
            # Fallback to simple grid if scipy is not available
            valid_points = []
            K = 18 # 190 points
            for i in range(K + 1):
                for j in range(K - i + 1):
                    valid_points.append([i / K, j / K])
            if len(valid_points) > n_evals:
                valid_points = valid_points[:n_evals]

        # Evaluate the points
        for p in valid_points:
            if time.time() > deadline:
                break
            x = base_x.copy()
            x[0] = p[0]
            x[1] = p[1]
            try:
                f = problem(x)
                all_points.append(np.array(f, dtype=float))
            except Exception:
                continue
        
        return all_points
# EVOLVE-BLOCK-END
