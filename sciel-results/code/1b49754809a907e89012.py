"""Bootstrap spawn: Greedy Linear Model Fit for Linear Slope f5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    Uses a greedy approach to determine gradient signs and move to the optimal vertex
    dimension by dimension, maximizing early progress (AOCC).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        best_x = np.zeros(self.dim)
        best_f = np.inf
        call_count = 0
        dim = self.dim
        
        # Step size for probing gradient signs
        probe_step = 0.1
        
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if time.time() > deadline:
                return best_f
            # Ensure x is a numpy array
            x = np.asarray(x, dtype=float)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Greedy Dimension Optimization
        # Current best point starts at center
        x_current = np.zeros(dim)
        
        # 1. Evaluate center
        f_current = evaluate(x_current)
        
        # Determine optimal value for each dimension sequentially
        # We maintain x_current as the best known point.
        # To probe dimension i, we evaluate x_current + probe_step * e_i
        # This tells us the sign of w_i.
        # Then we move x_current to the optimal bound for dimension i.
        
        for i in range(dim):
            if call_count >= self.budget or time.time() > deadline:
                break
            
            # Probe in positive direction of dimension i
            x_probe = x_current.copy()
            x_probe[i] += probe_step
            
            f_probe = evaluate(x_probe)
            
            # Determine sign of w_i
            # If f_probe > f_current, function increases in + direction => w_i > 0 => Opt is -5
            # If f_probe < f_current, function decreases in + direction => w_i < 0 => Opt is +5
            # Note: f_current is the value at x_current.
            # Since x_probe = x_current + probe_step * e_i, 
            # f_probe - f_current = w_i * probe_step.
            
            if f_probe > f_current:
                # w_i > 0, minimum is at -5
                optimal_val = -5.0
            else:
                # w_i < 0 (or 0, but BBOB usually has non-zero weights), minimum is at +5
                optimal_val = 5.0
            
            # Move to the optimal value for this dimension
            # Keep other dimensions at their current (already optimized) values
            x_new = x_current.copy()
            x_new[i] = optimal_val
            
            # Evaluate the new point
            f_current = evaluate(x_new)
            x_current = x_new

        # Phase 2: Fill remaining budget with best known point
        # The global minimum for a linear function is at the vertex.
        # We have found the vertex.
        # Re-evaluating it ensures b_i = f_opt for all remaining calls,
        # making n_i = 0.
        
        while call_count < self.budget and time.time() < deadline:
            evaluate(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
