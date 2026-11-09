"""
Exact Affine Model Recovery for BBOB f5 (Linear slope).

The function f5 is linear (affine). For a linear function on a hypercube,
the global minimum is always at a vertex.
By evaluating the function at the origin and along the axes, we can exactly
recover the gradient vector. Then we can analytically determine the optimal
vertex and jump to it in d+2 evaluations.

This minimizes the number of calls to find the global minimum, maximizing
the AOCC score.
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
        deadline = start_time + 290
        
        d = self.dim
        calls_made = 0
        best_val = float('inf')
        best_x = np.zeros(d)
        
        # For each dimension, evaluate both +5e_i and -5e_i to determine
        # the minimizing sign directly, without finite-difference gradient.
        # This gives better intermediate best values for AOCC.
        signs = np.zeros(d)  # +1 means x_i=5 is better, -1 means x_i=-5 is better
        
        for i in range(d):
            if time.time() > deadline:
                break
            
            # Evaluate +5 along dimension i
            x_plus = np.zeros(d)
            x_plus[i] = 5.0
            f_plus = func(x_plus)
            calls_made += 1
            if f_plus < best_val:
                best_val = f_plus
                best_x = x_plus.copy()
            
            if time.time() > deadline:
                break
            
            # Evaluate -5 along dimension i
            x_minus = np.zeros(d)
            x_minus[i] = -5.0
            f_minus = func(x_minus)
            calls_made += 1
            if f_minus < best_val:
                best_val = f_minus
                best_x = x_minus.copy()
            
            # Determine sign: if f_plus < f_minus, then v_i < 0, optimal x_i = +5
            # if f_plus > f_minus, then v_i > 0, optimal x_i = -5
            if f_plus < f_minus:
                signs[i] = 1.0
            else:
                signs[i] = -1.0
        
        # Construct the optimal vertex from the signs
        x_opt = np.zeros(d)
        for i in range(d):
            if signs[i] > 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = -5.0
        
        # Evaluate the optimal vertex
        if calls_made < self.budget:
            f_opt = func(x_opt)
            calls_made += 1
            if f_opt < best_val:
                best_val = f_opt
                best_x = x_opt.copy()
        
        # Stall: re-evaluate the best known point to fill the budget
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(best_x)
            calls_made += 1
        
        return best_x, best_val
# EVOLVE-BLOCK-END
