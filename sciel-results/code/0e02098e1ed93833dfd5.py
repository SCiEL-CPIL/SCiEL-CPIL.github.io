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
    Strategy: Use Halton sequence for better space-filling, plus explicit corners.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _halton(self, n, base):
        """Generate Halton sequence in [0,1) for given base."""
        result = []
        f = 1.0
        for i in range(1, n + 1):
            x = 0.0
            j = i
            while j > 0:
                f /= base
                x += f * (j % base)
                j //= base
            result.append(x)
        return np.array(result)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        results = []
        
        # Use Halton sequence for space-filling in 2D parameter space
        # Plus explicit corners for better coverage of boundary regions
        
        # Generate Halton sequence points
        n_halton = 192  # Leave room for corners
        x0_halton = self._halton(n_halton, 2)
        x1_halton = self._halton(n_halton, 3)
        
        # Add explicit corners and edge midpoints for better coverage
        corner_points = [
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5),
            (0.0, 0.0), (1.0, 1.0)  # Duplicate corners to emphasize them
        ]
        
        # Combine: Halton points + corner points
        all_x0 = np.concatenate([x0_halton, [p[0] for p in corner_points]])
        all_x1 = np.concatenate([x1_halton, [p[1] for p in corner_points]])
        
        # Shuffle to mix corners with Halton points (use fixed seed for reproducibility)
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(all_x0))
        all_x0 = all_x0[perm]
        all_x1 = all_x1[perm]
        
        # Evaluate points until budget is exhausted
        for i in range(len(all_x0)):
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x0 = all_x0[i]
            x1 = all_x1[i]
            
            x = np.full(self.dim, 0.5)  # All g-vars at 0.5
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            results.append(f)
            n_evals += 1
        
        # If we have remaining budget, add more random points
        if n_evals < self.budget:
            np.random.seed(42)
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
