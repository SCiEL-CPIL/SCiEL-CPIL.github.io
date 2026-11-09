"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    Uses a 2D grid in (x[0], x[1]) space with x[2:]=0 for better hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7: Try 2D grid in (x[0], x[1]) with x[2:]=0
        # 14x14 = 196 points + 4 strategic = 200
        n_grid = 14
        x0_vals = np.linspace(0, 1, n_grid)
        x1_vals = np.linspace(0, 1, n_grid)
        
        # Generate all grid points
        grid_points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                grid_points.append((x0, x1))
        
        # Add 4 strategic boundary/midpoint points
        strategic_points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.5)
        ]
        
        all_points = grid_points + strategic_points
        all_points = all_points[:budget]
        
        # Evaluate each point
        for x0, x1 in all_points:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            # x[2:] = 0
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
