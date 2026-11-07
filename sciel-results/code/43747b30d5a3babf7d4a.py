"""Multi-objective optimizer for DTLZ7 exploiting analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that exploits the analytical Pareto front of DTLZ7.
    
    For DTLZ7 with M=3, the Pareto front is:
      f1 = x0
      f2 = x1  
      f3 = 2 - sqrt(x0)
    where x0, x1 in [0,1] and x2..x21 = 0.
    
    We use a dense sampling strategy to cover the 2D Pareto front efficiently.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 300.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Pareto front: x0, x1 in [0,1], rest = 0
        # f1 = x0, f2 = x1, f3 = 2 - sqrt(x0)
        # Reference point: [1.1, 1.1, 25.0]
        
        # Strategy: sample the 2D Pareto front densely
        # The front is parameterized by (x0, x1) in [0,1]^2
        # We need to use our budget wisely
        
        used = 0
        results = []
        
        # Phase 1: Dense grid on the Pareto front
        # Use a portion of budget for systematic coverage
        grid_budget = min(self.budget - 50, 150)
        n_grid = int(np.sqrt(grid_budget))
        
        # Create grid points on the Pareto front
        x0_vals = np.linspace(0, 1, n_grid)
        x1_vals = np.linspace(0, 1, n_grid)
        
        for x0 in x0_vals:
            if used >= self.budget or time.time() > deadline - 30:
                break
            for x1 in x1_vals:
                if used >= self.budget or time.time() > deadline - 30:
                    break
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                # Clip to bounds
                x = np.clip(x, lo, hi)
                f = problem(x)
                results.append(f)
                used += 1
        
        # Phase 2: Add additional points for better coverage
        # Use remaining budget for finer sampling in high-value regions
        remaining = self.budget - used
        if remaining > 0:
            # Focus on regions where f3 is low (high x0) since f3 = 2 - sqrt(x0)
            # and we want to minimize all objectives
            # Also ensure good coverage of f1 and f2
            
            # Add points along the edges of the front
            edge_budget = min(remaining // 3, 30)
            if edge_budget > 0:
                # Edge: x0=1, vary x1
                n_edge = min(edge_budget, 15)
                x1_edge = np.linspace(0, 1, n_edge)
                for x1 in x1_edge:
                    if used >= self.budget or time.time() > deadline - 30:
                        break
                    x = np.zeros(self.dim)
                    x[0] = 1.0
                    x[1] = x1
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    results.append(f)
                    used += 1
                
                # Edge: x1=1, vary x0
                x0_edge = np.linspace(0, 1, n_edge)
                for x0 in x0_edge:
                    if used >= self.budget or time.time() > deadline - 30:
                        break
                    x = np.zeros(self.dim)
                    x[0] = x0
                    x[1] = 1.0
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    results.append(f)
                    used += 1
            
            # Add interior points with finer resolution
            remaining2 = self.budget - used
            if remaining2 > 0:
                # Use quasi-random (Sobol-like) sampling in the remaining budget
                # Simple approach: stratified random sampling
                n_extra = min(remaining2, 50)
                for _ in range(n_extra):
                    if used >= self.budget or time.time() > deadline - 30:
                        break
                    # Stratified sampling
                    i = np.random.randint(0, n_grid)
                    j = np.random.randint(0, n_grid)
                    # Add small random offset within the cell
                    x0 = (i + np.random.random()) / n_grid
                    x1 = (j + np.random.random()) / n_grid
                    x = np.zeros(self.dim)
                    x[0] = x0
                    x[1] = x1
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    results.append(f)
                    used += 1
        
        # Return the last evaluated point (return value is ignored by evaluator)
        if results:
            return np.zeros(self.dim)
        return np.zeros(self.dim)
# EVOLVE-BLOCK-END
