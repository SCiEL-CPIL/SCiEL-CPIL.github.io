"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-slope optimizer: identify optimal corner via gradient estimation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Phase 1: Estimate gradient signs using coordinate probes
        # For a linear function f(x) = c + w·x, we need sign of w_i for each i
        # Use a reference point and probe each coordinate in both directions
        
        # Start at center
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1
        
        # Probe each coordinate to determine gradient sign
        # Use a small step to stay well within bounds
        step = 4.0  # probe from center to near-boundary
        
        grad_signs = np.zeros(dim)
        for i in range(dim):
            if time.time() > deadline - 5:
                break
            # Probe positive direction
            x_pos = x0.copy()
            x_pos[i] = step
            f_pos = func(x_pos)
            calls += 1
            
            # Probe negative direction
            x_neg = x0.copy()
            x_neg[i] = -step
            f_neg = func(x_neg)
            calls += 1
            
            # For linear function: f_pos - f0 ≈ w_i * step, f0 - f_neg ≈ w_i * step
            # If f_pos > f0, w_i > 0, so minimum is at x_i = -5
            # If f_pos < f0, w_i < 0, so minimum is at x_i = +5
            if f_pos > f0:
                grad_signs[i] = -1  # go negative
            elif f_pos < f0:
                grad_signs[i] = 1   # go positive
            else:
                # Flat in this direction, try the other probe
                if f_neg > f0:
                    grad_signs[i] = 1   # go positive (f_neg > f0 means w_i < 0... wait)
                    # f_neg = f0 + w_i*(-step) = f0 - w_i*step
                    # If f_neg > f0, then -w_i*step > 0, so w_i < 0, go positive
                    grad_signs[i] = 1
                elif f_neg < f0:
                    # f_neg < f0 means -w_i*step < 0, so w_i > 0, go negative
                    grad_signs[i] = -1
                else:
                    grad_signs[i] = 0  # flat, doesn't matter
            
            # Check budget
            if calls >= budget:
                break
        
        # Phase 2: Jump to the optimal corner
        x_corner = grad_signs * 5.0  # each coordinate is -5 or +5
        # Clip to be safe
        x_corner = np.clip(x_corner, -5, 5)
        
        f_corner = func(x_corner)
        calls += 1
        
        best_x = x_corner
        best_f = f_corner
        
        # Phase 3: Use remaining budget for verification and potential refinement
        # For a truly linear function, the corner is optimal. But let's verify
        # and use any remaining budget wisely.
        
        # Check if we should try nearby points in case of non-linearity or noise
        remaining = budget - calls
        if remaining > 0 and time.time() < deadline - 2:
            # Try a few random points to verify we're at the optimum
            rng = np.random.default_rng(42)
            for _ in range(min(remaining, 50)):
                if time.time() > deadline - 2:
                    break
                # Sample near the corner or randomly
                if rng.random() < 0.5:
                    # Near the corner
                    x_try = x_corner + rng.normal(0, 0.5, dim)
                else:
                    # Random in box
                    x_try = rng.uniform(-5, 5, dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls += 1
                if f_try < best_f:
                    best_x = x_try.copy()
                    best_f = f_try
        
        return best_x, best_f
# EVOLVE-BLOCK-END
