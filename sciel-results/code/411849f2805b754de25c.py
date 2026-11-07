"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizes a linear function on a hypercube using a 6-point simplex.
    
    Strategy:
    1. Evaluate at the origin (center).
    2. Evaluate at the 5 axis vertices (distance 5.0) to determine the gradient.
       Using the maximal step size (5.0) maximizes the potential function value drop 
       at each probe step, improving the 'best-so-far' metric (b_i) early on.
    3. Compute the exact optimal vertex from the gradient signs.
    4. Evaluate at the optimal vertex.
    5. Fill remaining budget with evaluations of the optimal vertex.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # 1. Evaluate at origin
        x_origin = np.zeros(dim)
        f_origin = func(x_origin)
        
        # 2. Probe along each axis with maximal step size
        h = 5.0
        g = np.zeros(dim)
        x_probe = np.zeros(dim)
        
        for i in range(dim):
            x_probe[i] = h
            f_probe = func(x_probe)
            g[i] = (f_probe - f_origin) / h
            x_probe[i] = 0.0  # Reset for next iteration
            
        # 3. Determine optimal vertex
        # x_i = -5 * sign(g_i) minimizes g^T x
        x_opt = -5.0 * np.sign(g)
        
        # 4. Evaluate at optimal vertex
        f_opt = func(x_opt)
        
        # 5. Consume remaining budget
        calls_made = 1 + dim + 1
        remaining = budget - calls_made
        if remaining > 0:
            for _ in range(remaining):
                _ = func(x_opt)
                
        return x_opt, f_opt
# EVOLVE-BLOCK-END
