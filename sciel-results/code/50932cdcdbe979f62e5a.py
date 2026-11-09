"""
Optimized Affine Recovery for BBOB f5.

This program exploits the affine nature of the f5 function (Linear Slope)
to determine the global minimum in exactly 7 evaluations:
1. Evaluation at the origin to obtain the offset.
2. Five evaluations along the coordinate axes to determine the gradient.
3. One evaluation at the analytically derived optimal vertex.

This strategy minimizes the Area Over the Curve Cost (AOCC) by reaching the
global minimum as early as possible in the budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety deadline: 300s limit, keep 20s buffer for overhead
        deadline = start_time + 280
        
        d = self.dim
        calls_made = 0
        best_val = float('inf')
        best_x = np.zeros(d)
        
        # 1. Evaluate at the origin
        x_origin = np.zeros(d)
        f_origin = func(x_origin)
        calls_made += 1
        best_val = f_origin
        best_x = x_origin.copy()
        
        # 2. Evaluate along positive axes to recover gradient
        # f(x) = f(0) + grad . x
        # grad[i] = (f(5*e_i) - f(0)) / 5
        grad = np.zeros(d)
        axis_evals = []
        
        for i in range(d):
            # Time check to prevent infinite loops or timeout
            if time.time() > deadline:
                break
            
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            axis_evals.append(fi)
            
            # Update best
            if fi < best_val:
                best_val = fi
                best_x = xi.copy()
                
            # Compute gradient component
            grad[i] = (fi - f_origin) / 5.0
            
            # If we have enough gradient info, we could potentially stop early
            # but we need all components to define the vertex.
            
        # 3. Determine the optimal vertex
        # For f(x) = c + v^T x on [-5, 5]^d:
        # x*_i = -5 if v_i > 0
        # x*_i =  5 if v_i < 0
        # x*_i =  0 if v_i == 0 (arbitrary)
        
        # Vectorized sign determination
        # np.sign returns -1, 0, 1
        # We want -5 if grad > 0, 5 if grad < 0.
        # So x_opt = -5 * sign(grad)
        # If grad is 0, sign is 0, x is 0.
        
        x_opt = -5.0 * np.sign(grad)
        
        # Ensure x_opt is strictly within bounds (handling floating point edge cases)
        # np.sign can produce -0.0, but -5.0 * -0.0 is 0.0.
        # If grad is extremely small but positive, sign is 1, x is -5.
        # If grad is extremely small but negative, sign is -1, x is 5.
        
        # 4. Evaluate the optimal vertex
        if calls_made < self.budget and time.time() < deadline:
            f_opt = func(x_opt)
            calls_made += 1
            
            if f_opt < best_val:
                best_val = f_opt
                best_x = x_opt.copy()
                
        # If for some reason we didn't reach the optimal point (e.g. timeout),
        # we return the best found so far.
        
        return best_x, best_val
# EVOLVE-BLOCK-END
