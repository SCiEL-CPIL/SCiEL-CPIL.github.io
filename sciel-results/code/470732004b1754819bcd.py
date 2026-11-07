"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-aware optimizer: estimate gradient via boundary probes, jump to optimal vertex."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        # Phase 1: Estimate gradient signs using 6 evaluations (origin + 5 boundary probes)
        # Use h=5.0 (box boundary) to maximize the magnitude of the signal for sign determination
        # and to maximize the potential improvement in 'best so far' if the probe is in the correct direction.
        h = 5.0
        x0 = np.zeros(dim)
        f0 = func(x0)
        
        best_val = f0
        best_x = x0.copy()
        
        # grad_signs[i] will be 1 if c_i > 0, -1 if c_i < 0
        # We need to determine the direction of the minimum.
        # f(x) = c^T x + b.
        # If c_i > 0, min is at x_i = -5.
        # If c_i < 0, min is at x_i = +5.
        
        c_signs = np.zeros(dim)
        
        for i in range(dim):
            xi = x0.copy()
            xi[i] += h
            fi = func(xi)
            
            # Update best so far
            if fi < best_val:
                best_val = fi
                best_x = xi.copy()
            
            # Determine sign of c_i
            # fi - f0 = h * c_i
            # If fi > f0, c_i > 0 -> optimal x_i is -5
            # If fi < f0, c_i < 0 -> optimal x_i is +5
            if fi > f0:
                c_signs[i] = 1.0
            elif fi < f0:
                c_signs[i] = -1.0
            else:
                c_signs[i] = 0.0 # Flat, doesn't matter, default to 0 or either side

        # Phase 2: Construct the optimal vertex
        x_opt = np.zeros(dim)
        for i in range(dim):
            if c_signs[i] > 0:
                x_opt[i] = -5.0
            elif c_signs[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0
        
        # Phase 3: Evaluate the optimal vertex
        f_opt_est = func(x_opt)
        
        if f_opt_est < best_val:
            best_val = f_opt_est
            best_x = x_opt.copy()
        else:
            # This shouldn't happen for a linear function if the logic is correct, 
            # but just in case of numerical issues or non-linearities (though task says linear)
            # we keep the best found so far.
            x_opt = best_x.copy()
            f_opt_est = best_val

        # Phase 4: Spend remaining budget at the best known point
        # We have used 1 + dim + 1 = dim + 2 calls.
        calls_used = dim + 2
        remaining_calls = budget - calls_used
        
        # Ensure we don't exceed budget
        for _ in range(max(0, remaining_calls)):
            if time.time() - start_time > deadline:
                break
            func(x_opt)
        
        return x_opt, f_opt_est
# EVOLVE-BLOCK-END
