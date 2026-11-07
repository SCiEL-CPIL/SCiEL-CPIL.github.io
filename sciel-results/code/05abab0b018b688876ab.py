"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear model extraction followed by (1+1)-ES polish."""

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

        # Phase 1: Estimate gradient using linear model
        # Evaluate at center (0,0,0,0,0)
        x_center = np.zeros(self.dim)
        f_center = func(x_center)
        update_best(x_center, f_center)
        
        # Check deadline
        if time.time() > deadline:
            return best_x, best_f

        # Use large step to maximize signal-to-noise ratio for gradient estimation
        # Step of 5.0 reaches the boundary from the center
        step = 5.0
        gradients = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
            x_probe = x_center.copy()
            x_probe[i] += step
            f_probe = func(x_probe)
            update_best(x_probe, f_probe)
            gradients[i] = (f_probe - f_center) / step
        
        # Determine the vertex of the box that minimizes the linear model
        # f(x) ~ f_center + sum(gradients[i] * x[i])
        # To minimize, if gradient[i] > 0, we want x[i] to be as small as possible (-5)
        # If gradient[i] < 0, we want x[i] to be as large as possible (5)
        x_vertex = np.zeros(self.dim)
        for i in range(self.dim):
            if gradients[i] > 0:
                x_vertex[i] = -5.0
            else:
                x_vertex[i] = 5.0
        
        # Evaluate the candidate optimal vertex
        f_vertex = func(x_vertex)
        update_best(x_vertex, f_vertex)
        
        # Phase 2: Polish with (1+1)-ES
        # This helps if the function has slight non-linearities or if the gradient
        # estimation was affected by numerical issues, or if the true optimum
        # is not exactly at the vertex due to some other factor.
        x = x_vertex.copy()
        fx = f_vertex
        sigma = 0.5 # Start with a moderate step size for polishing
        
        # Calculate calls used so far: 1 (center) + dim (probes) + 1 (vertex)
        calls_used = 1 + self.dim + 1
        remaining_calls = self.budget - calls_used
        
        rng = np.random.default_rng(42)
        
        for _ in range(remaining_calls):
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            # Clip to box
            y = np.clip(y, -5.0, 5.0)
            
            fy = func(y)
            update_best(y, fy)
            
            # Adapt step size
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1 # Expand slightly
            else:
                sigma *= 0.9 # Shrink
            
            # If sigma becomes too small, we are likely converged
            if sigma < 1e-6:
                # We can break early if we are very confident
                # But let's just let it run out of budget or time
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
