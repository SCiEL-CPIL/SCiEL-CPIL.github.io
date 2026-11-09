import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        x_center = np.zeros(dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        
        # Estimate gradient signs using steps to the boundary
        # For a linear function f(x) = a^T x + b, the minimum on [-5, 5]^dim
        # is at a vertex. The sign of a_i determines if x_i should be 5 or -5.
        # We test f(5 e_i) vs f(0) to determine the sign of a_i.
        
        signs = np.zeros(dim)
        calls_used = 1
        
        for i in range(dim):
            if time.time() > deadline:
                break
            
            x_plus = x_center.copy()
            x_plus[i] = 5.0
            
            f_plus = func(x_plus)
            calls_used += 1
            
            if f_plus < best_f:
                best_f = f_plus
                best_x = x_plus.copy()
            
            # Determine sign of gradient component i
            # If f(5 e_i) < f(0), gradient is negative, so min is at +5
            # If f(5 e_i) > f(0), gradient is positive, so min is at -5
            # If equal, either is fine, default to -5 (or +5, doesn't matter)
            if f_plus < f_center:
                signs[i] = 1.0  # Gradient negative, go to +5
            else:
                signs[i] = -1.0   # Gradient positive, go to -5
                
        # Construct the vertex corresponding to the estimated gradient signs
        x_vertex = signs * 5.0
        
        f_vertex = func(x_vertex)
        calls_used += 1
        
        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()
            
        # For a linear function, this vertex is the global minimum.
        # No further refinement is needed.
        
        return best_x, best_f
