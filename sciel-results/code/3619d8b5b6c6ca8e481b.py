"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Coordinate Descent Optimizer for Sphere Function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Deadline guard: 330s limit, keep 30s margin
        start_time = time.time()
        deadline = start_time + 300.0
        
        # Start at center (0,0,0,0,0) as a strong prior for the minimum
        # The problem states the minimizer is in [-4, 4]^5, so 0 is a good start.
        x = np.zeros(self.dim)
        fx = func(x)
        calls_made = 1
        
        # Initial step size for coordinate descent
        # Since the box is [-5, 5], a step of 1.0 is reasonable to start
        step = 1.0
        
        # We will iterate until we run out of budget or time
        # Coordinate descent cycles through dimensions
        dim_idx = 0
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            d = dim_idx % self.dim
            dim_idx += 1
            
            # Try moving in positive direction
            x_plus = x.copy()
            x_plus[d] += step
            if x_plus[d] > 5.0: x_plus[d] = 5.0
            
            fx_plus = func(x_plus)
            calls_made += 1
            
            if fx_plus < fx:
                x = x_plus
                fx = fx_plus
                # Increase step size if we made progress (acceleration)
                # But be careful not to overshoot too much. 
                # For Sphere, if we are far, larger steps are good.
                # If we are close, we need smaller steps.
                # A simple heuristic: increase step, but if we hit boundary, decrease.
                # Actually, for Sphere, the gradient magnitude decreases as we approach 0.
                # Let's use a standard line search-like approach or just adaptive step.
                # Let's try: if we moved, keep step or increase slightly. 
                # However, if we are near the optimum, we need to shrink.
                # Let's use a simple rule: if we improve, step *= 1.1. 
                # If we don't improve in either direction, step *= 0.5.
                step *= 1.1
                # Cap step to avoid huge jumps
                if step > 5.0:
                    step = 5.0
            else:
                # Try moving in negative direction
                x_minus = x.copy()
                x_minus[d] -= step
                if x_minus[d] < -5.0: x_minus[d] = -5.0
                
                fx_minus = func(x_minus)
                calls_made += 1
                
                if fx_minus < fx:
                    x = x_minus
                    fx = fx_minus
                    step *= 1.1
                    if step > 5.0:
                        step = 5.0
                else:
                    # No improvement in either direction for this coordinate
                    # Reduce step size
                    step *= 0.5
                    # If step becomes very small, we might be converged
                    if step < 1e-6:
                        # Try to reset step for next dimension? 
                        # Or just continue with small steps.
                        # Given Sphere, if step is tiny, we are likely at optimum.
                        # But we have budget, let's keep going just in case of noise/precision.
                        pass

        return x, fx
# EVOLVE-BLOCK-END
