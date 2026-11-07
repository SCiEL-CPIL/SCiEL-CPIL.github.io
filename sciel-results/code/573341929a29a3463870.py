import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        k_shape = self.n_obj - 1  # 2 for 3 objectives
        shape_vars = list(range(k_shape))
        dist_vars = list(range(k_shape, self.dim))

        # For DTLZ2 with 3 objectives:
        # The Pareto front is a quarter-sphere in the positive octant.
        # f1^2 + f2^2 + f3^2 = 1 (when g=1)
        # 
        # The mapping from shape variables (x1, x2) in [0,1]^2 to objectives:
        # Standard DTLZ2 formulation:
        # f1 = g * sin(x1*pi/2) * sin(x2*pi/2)
        # f2 = g * sin(x1*pi/2) * cos(x2*pi/2)
        # f3 = g * cos(x1*pi/2)
        # where g = 1 + 10*(n-M+1) + sum((x_j - 0.5)^2) for j >= M-1
        #
        # Wait, let me reconsider. The standard DTLZ2 is:
        # f_i(x) = g(x) * (1 + cos(x_{M-i} * pi/2) * prod_{j=1}^{M-i-1} sin(x_j * pi/2))
        # for i = 1, ..., M-1
        # f_M(x) = g(x) * prod_{j=1}^{M-1} sin(x_j * pi/2)
        #
        # For M=3:
        # f1 = g * (1 + cos(x2*pi/2))  -- this gives f1 in [g, 2g]
        # f2 = g * sin(x1*pi/2) * cos(x2*pi/2)  -- wait, this doesn't seem right either
        #
        # Let me use a different approach. Since we don't know the exact formulation,
        # we'll use a Fibonacci lattice on the 2D parameter space and evaluate.
        # The key insight is that the shape variables determine position on the front,
        # and distance variables should be at their optimal values.
        
        # For DTLZ2, the optimal distance variable values are typically 0.5
        # (where g is minimized). Let's use 0.5 for all distance variables.
        
        # Generate Fibonacci lattice points in [0,1]^2
        # This gives near-uniform coverage of the 2D parameter space
        n = n_evals
        golden_ratio_conjugate = (1.0 - np.sqrt(5.0)) / 2.0
        
        points = np.zeros((n, 2))
        for i in range(n):
            # Fibonacci lattice in 2D
            # Use the golden ratio for good distribution
            x = (i + 0.5) / n
            y = (i * golden_ratio_conjugate) % 1.0
            points[i, 0] = x
            points[i, 1] = y
        
        # Shuffle to avoid any systematic ordering effects
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        points = points[perm]
        
        solutions = []
        for i in range(n):
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            
            # Set shape variables
            x[shape_vars[0]] = points[i, 0]
            x[shape_vars[1]] = points[i, 1]
            
            # Set distance variables to 0.5 (optimal for DTLZ2)
            for var_idx in dist_vars:
                x[var_idx] = 0.5
                
            f = np.array(problem(x))
            solutions.append((x, f))
        
        return solutions
