"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate Descent with adaptive step sizes for Sphere minimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(123)

        # Start at center (0)
        x = np.zeros(dim)
        fx = func(x)
        best_f = fx
        calls_made = 1

        # Initialize step sizes
        step_sizes = np.full(dim, 1.0)

        # Coordinate descent loop
        # We have 499 calls left.
        # Cycle through dimensions.
        # For each dimension, try a step.
        # If successful, keep step size (or increase slightly).
        # If failed, decrease step size.
        
        i = 0
        while calls_made < budget and time.time() < deadline:
            d = i % dim
            i += 1
            
            # Try positive step
            x_plus = x.copy()
            x_plus[d] += step_sizes[d]
            x_plus = np.clip(x_plus, -5, 5)
            
            f_plus = func(x_plus)
            calls_made += 1
            if calls_made >= budget or time.time() >= deadline:
                break
                
            if f_plus < fx:
                x = x_plus
                fx = f_plus
                if fx < best_f:
                    best_f = fx
                # Increase step size slightly if we made progress
                step_sizes[d] *= 1.1
            else:
                # Try negative step
                x_minus = x.copy()
                x_minus[d] -= step_sizes[d]
                x_minus = np.clip(x_minus, -5, 5)
                
                f_minus = func(x_minus)
                calls_made += 1
                if calls_made >= budget or time.time() >= deadline:
                    break
                    
                if f_minus < fx:
                    x = x_minus
                    fx = f_minus
                    if fx < best_f:
                        best_f = fx
                    step_sizes[d] *= 1.1
                else:
                    # No progress in either direction, reduce step size
                    step_sizes[d] *= 0.7
                    # If step size becomes too small, reset to a random value to escape local minima?
                    # For Sphere, there are no local minima, so reducing is fine until convergence.
                    # But if we get stuck due to precision, we might need to restart or randomize.
                    # However, Sphere is convex. Coordinate descent converges.
                    # If step size is very small, we are likely at the minimum for that coordinate.
                    if step_sizes[d] < 1e-6:
                         step_sizes[d] = 0.1
                         # Randomize direction? No, just try again with larger step.
                         # Actually, for Sphere, if we are at the min, any step increases f.
                         # So we will keep reducing step size.
                         # This is inefficient if we are already at the min.
                         # Let's add a check: if we haven't improved in last K steps, do a random jump?
                         # But for Sphere, random jump is bad.
                         # Just let it converge. The cost is low.

        return x, fx
# EVOLVE-BLOCK-END
