"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizes the BBOB f5 (Linear slope) function by sequentially
    determining the optimal direction for each dimension.
    
    Strategy:
    1. Start at the center (0, ..., 0).
    2. For each dimension i:
       a. Probe the positive direction (+5) from the current best point.
       b. Estimate the gradient component for dimension i.
       c. Determine the optimal bound (-5 or +5) for dimension i.
       d. If the optimal bound is different from the probe direction,
          evaluate the new point. Otherwise, reuse the probe value.
       e. Update the current best point to include this optimized dimension.
    
    This ensures that the best-so-far value improves (or stays same) 
    at every step, maximizing the Area Under the Convergence Curve (AOCC).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 290  # Safety margin for 300s limit
        
        # 1. Evaluate center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        f_best = f_center
        x_best = x_center.copy()
        
        # 2. Probe each axis to determine gradient signs
        # For linear f(x) = g^T x + c, probing 5*e_i gives f(5e_i) = 5*g_i + c
        # So g_i = (f(5e_i) - f(0)) / 5
        opt_vertex = np.zeros(dim)
        for i in range(dim):
            if time.time() > deadline:
                break
            
            x_probe = np.zeros(dim)
            x_probe[i] = 5.0
            f_probe = func(x_probe)
            
            grad_i = (f_probe - f_center) / 5.0
            
            if grad_i > 0:
                opt_vertex[i] = -5.0
            else:
                opt_vertex[i] = 5.0
        
        # 3. Evaluate the optimal vertex
        if time.time() > deadline:
            return x_best, f_best
        
        f_opt = func(opt_vertex)
        x_best = opt_vertex
        f_best = f_opt
        
        # 4. Fill remaining budget with the optimal point (or nearby points for safety)
        # The AOCC uses best-so-far, so repeating the same point doesn't change b_i
        # We just need to make calls to fill the budget without exceeding it
        calls_made = 1 + dim + 1  # center + probes + optimal vertex
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            # Evaluate the optimal point again (or we could do a small perturbation,
            # but for a linear function, the vertex is the global min)
            # To be safe and potentially help with numerical issues, evaluate opt_vertex
            f_val = func(opt_vertex)
            if f_val < f_best:
                f_best = f_val
            calls_made += 1
        
        return x_best, f_best
# EVOLVE-BLOCK-END
