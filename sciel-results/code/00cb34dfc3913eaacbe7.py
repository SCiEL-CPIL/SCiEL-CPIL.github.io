"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear model extraction followed by robust (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s, we aim for 290s
        deadline = start_time + 290.0
        
        best_x = None
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x

        # Phase 1: Estimate gradient using linear model at h=4.0
        # Larger step size gives more robust sign determination for linear functions.
        
        x_center = np.zeros(self.dim)
        f_center = func(x_center)
        update_best(x_center, f_center)
        
        if time.time() > deadline:
            return best_x, best_f
        
        h = 4.0
        gradients = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
            x_probe = x_center.copy()
            x_probe[i] += h
            f_probe = func(x_probe)
            update_best(x_probe, f_probe)
            
            # Linear approximation: f(x) ~ f_center + sum(w_i * x_i)
            # w_i = (f_probe - f_center) / h
            gradients[i] = (f_probe - f_center) / h
        
        # Determine the vertex of the box that minimizes the linear model
        x_vertex = np.zeros(self.dim)
        for i in range(self.dim):
            if gradients[i] > 0:
                x_vertex[i] = -5.0
            else:
                x_vertex[i] = 5.0
        
        # Evaluate the candidate optimal vertex
        f_vertex = func(x_vertex)
        update_best(x_vertex, f_vertex)
        
        # Phase 2: Local verification and polish around the best point
        # For a linear function, the optimum is at a vertex. We do a local search
        # with small steps to verify and handle any edge cases.
        x = best_x.copy()
        fx = best_f
        
        # Start with a small step size for local exploration
        sigma = 0.5
        
        calls_used = 1 + self.dim + 1
        remaining_calls = self.budget - calls_used
        
        rng = np.random.default_rng(42)
        
        for _ in range(remaining_calls):
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5.0, 5.0)
            
            fy = func(y)
            update_best(y, fy)
            
            # Adapt step size
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1
            else:
                sigma *= 0.95
            
            if sigma < 1e-12:
                sigma = 1e-12

        return best_x, best_f
# EVOLVE-BLOCK-END
