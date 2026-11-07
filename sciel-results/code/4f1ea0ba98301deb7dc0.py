"""Rewritten program: Exact linear optimization using gradient estimation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer that exploits the linearity of the BBOB f5 function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Safety deadline: 290 seconds out of 300 allowed
        start_time = time.time()
        deadline = start_time + 290.0
        
        d = self.dim
        
        # Step 1: Evaluate at the origin
        x0 = np.zeros(d)
        f0 = func(x0)
        
        # Step 2: Estimate the gradient vector c
        # f(x) = c^T x + b
        # f(0) = b
        # f(delta * e_i) = c_i * delta + b
        # c_i = (f(delta * e_i) - f(0)) / delta
        # Using delta=5.0 maximizes signal strength for linear gradient estimation
        # and ensures probes are at the box boundaries, providing the largest possible
        # initial drop in function value for components with negative gradients.
        delta = 5.0
        c = np.zeros(d)
        
        for i in range(d):
            if time.time() > deadline:
                # If we run out of time, return the best we have (origin)
                # This is extremely unlikely for 5 calls
                return x0, f0
                
            probe = np.zeros(d)
            probe[i] = delta
            f_probe = func(probe)
            c[i] = (f_probe - f0) / delta
            
        # Step 3: Determine the optimal vertex
        # The minimum of a linear function on a hypercube is at a vertex.
        # x_i^* = 5 if c_i < 0, -5 if c_i > 0
        x_opt = np.zeros(d)
        for i in range(d):
            if c[i] < 0:
                x_opt[i] = 5.0
            elif c[i] > 0:
                x_opt[i] = -5.0
            else:
                x_opt[i] = 0.0
                
        # Step 4: Evaluate the optimal vertex
        if time.time() > deadline:
            return x0, f0
            
        f_opt = func(x_opt)
        
        return x_opt, f_opt
# EVOLVE-BLOCK-END
