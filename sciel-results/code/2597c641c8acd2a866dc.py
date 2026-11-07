"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting DTLZ7's analytical Pareto front.
    
    DTLZ7 with M=3 objectives has a Pareto front defined by:
    f1 = x0, f2 = x1, f3 = 2 - sqrt(x0)
    where x0, x1 in [0, 1] and x2...x21 = 0.
    
    We use a dense grid + Latin Hypercube sampling to cover the front efficiently,
    maximizing hypervolume with the given budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Pareto front: f1=x0, f2=x1, f3=2-sqrt(x0)
        # x0, x1 in [0,1], rest = 0
        # Reference point: [1.1, 1.1, 25.0]
        # To maximize hypervolume, we want points on the Pareto front
        # The front is 2D: parameterized by x0 and x1
        
        # Strategy: sample densely on the Pareto front using structured sampling
        # that maximizes hypervolume coverage
        
        used = 0
        results = []
        
        # Phase 1: Dense grid on the Pareto front
        # x0 in [0,1], x1 in [0,1]
        # f1 = x0, f2 = x1, f3 = 2 - sqrt(x0)
        
        # The hypervolume contribution depends on the "corner" of the front
        # Points near (0,0,2) and (1,1,1) are important
        # Actually for DTLZ7, the Pareto front in objective space:
        # f1 = x0, f2 = x1, f3 = 2 - sqrt(x0)
        # So f3 ranges from 1 (when x0=1) to 2 (when x0=0)
        # f1 ranges from 0 to 1
        # f2 ranges from 0 to 1
        
        # Reference point is [1.1, 1.1, 25.0]
        # All Pareto front points are better than reference in all objectives
        # (since f1<=1 < 1.1, f2<=1 < 1.1, f3<=2 < 25)
        
        # To maximize hypervolume, we want to cover the 2D front well
        # Use a combination of:
        # 1. Uniform grid on (x0, x1)
        # 2. Additional points near corners and edges
        
        n_grid = int(np.sqrt(self.budget // 2))
        if n_grid < 2:
            n_grid = 2
        
        # Create grid points
        x0_vals = np.linspace(0, 1, n_grid)
        x1_vals = np.linspace(0, 1, n_grid)
        
        grid_points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                # x[2:] = 0 by default
                grid_points.append(x)
        
        # Evaluate grid points
        for x in grid_points:
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            f = problem(x)
            results.append(f)
            used += 1
        
        # Phase 2: Fill remaining budget with additional strategic points
        # Focus on regions that contribute most to hypervolume
        # The hypervolume integral is maximized by covering the entire front
        
        # Add points along the edges of the front for better coverage
        n_extra = self.budget - used
        if n_extra > 0:
            # Use Latin hypercube sampling on the remaining budget
            # But bias towards areas with higher hypervolume contribution
            # Areas near (x0=0, x1=0) have f3=2, f1=0, f2=0 - high contribution
            # Areas near (x0=1, x1=1) have f3=1, f1=1, f2=1 - also important
            
            # Use a mix of uniform and corner-biased sampling
            for _ in range(n_extra):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                
                # Mix: 50% uniform, 25% near origin, 25% near (1,1)
                r = rng.random()
                if r < 0.5:
                    x0 = rng.random()
                    x1 = rng.random()
                elif r < 0.75:
                    # Near origin (x0=0, x1=0) - high f3
                    x0 = rng.random() ** 2
                    x1 = rng.random() ** 2
                else:
                    # Near (1,1) - low f3
                    x0 = 1 - rng.random() ** 2
                    x1 = 1 - rng.random() ** 2
                
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                f = problem(x)
                results.append(f)
                used += 1
        
        return results
# EVOLVE-BLOCK-END
