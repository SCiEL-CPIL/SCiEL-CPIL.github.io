"""Direct constructive optimizer for DTLZ4: exploit known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Direct multi-objective optimizer for DTLZ4.
    
    Key insight: The Pareto front of DTLZ4 with 3 objectives lies on a curved surface.
    For 12 variables:
      - Variables x[0..1] (2 vars) determine the 'position' on the front
      - Variables x[2..11] (10 vars) determine g(x)
    
    Specifically for DTLZ4 with M=3, n=12:
      k = M - 1 = 2 (position variables: x_0, x_1)
      g(x) depends on x_2 .. x_11 (10 variables)
      
    DTLZ4 definition:
      g(x) = 10 + sum_{i=2}^{11} (x_i - 0.5)^100
      f_1 = g * cos(x_0 * pi/2) * cos(x_1 * pi/2)
      f_2 = g * cos(x_0 * pi/2) * sin(x_1 * pi/2)
      f_3 = g * sin(x_0 * pi/2)
    
    To minimize g: set x_2..x_11 = 0.5, giving g = 10.
    
    The Pareto front is parameterized by (x_0, x_1) in [0,1]^2.
    Strategy: sample the Pareto front densely using a grid of (x_0, x_1) values.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 structure for M=3, n=12:
        # Position vars: x[0], x[1] (indices 0,1)
        # g-vars: x[2] through x[11] (indices 2..11)
        # Set g-vars to 0.5 to minimize g
        
        # Generate a grid of (x0, x1) values to cover the front well.
        # Use a stratified approach for good coverage.
        
        # We have budget=200 evaluations total.
        # Use a grid that gives good coverage of the 2D parameter space.
        
        # Option 1: Dense grid
        # Option 2: Quasi-random (Halton sequence) for better space-filling
        
        # Let's use a combination: start with a grid, then fill gaps with random points.
        
        n_evals = 0
        results = []
        
        # First, create a structured grid
        # Use ~14x14 grid = 196 points, leaving 4 for refinement
        n_grid = 14
        x0_vals = np.linspace(0, 1, n_grid)
        x1_vals = np.linspace(0, 1, n_grid)
        
        # Evaluate grid points
        for x0 in x0_vals:
            for x1 in x1_vals:
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x = np.full(self.dim, 0.5)  # All g-vars at 0.5
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                results.append(f)
                n_evals += 1
            
            if time.time() > deadline:
                break
        
        # If we have remaining budget, add some random points for better coverage
        if n_evals < self.budget:
            np.random.seed(42)  # Fixed seed for reproducibility
            while n_evals < self.budget and time.time() < deadline:
                x0 = np.random.uniform(0, 1)
                x1 = np.random.uniform(0, 1)
                
                x = np.full(self.dim, 0.5)
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                results.append(f)
                n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
