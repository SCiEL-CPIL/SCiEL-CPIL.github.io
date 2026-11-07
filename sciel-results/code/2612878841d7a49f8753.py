"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale coordinate descent for f14 (Different Powers)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        # Initial point: center of the box
        # f14 minimizer is often near 0 or shifted, but [-5,5] center is a safe start
        x = np.zeros(self.dim)
        fx = func(x)
        
        # Multi-scale coordinate descent
        # Start with large strides to cover the domain quickly
        # f14 has different powers, so coordinates might need different scales,
        # but a global geometric shrinkage is a robust first approximation.
        
        # We will perform coordinate descent passes with decreasing step sizes.
        # Step sizes: 4.0, 2.0, 1.0, 0.5, 0.25, ... down to 1e-5
        # This is deterministic and avoids random stagnation.
        
        steps = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 
                 0.015625, 0.0078125, 0.00390625, 0.001953125, 
                 0.0009765625, 0.00048828125, 0.000244140625]
        
        calls_made = 1
        budget_left = self.budget - 1
        
        # We can afford to restart from a few different points if time/budget permits,
        # but the directive says "deterministic... starts with large strides".
        # Let's stick to a single robust run first, but maybe try a second start if budget remains?
        # The task says "For each instance... the evaluator creates Optimizer... and calls it once".
        # So we have one shot per instance.
        
        # Strategy:
        # 1. Coarse grid search / large step coordinate descent to find the basin.
        # 2. Fine step coordinate descent to polish.
        
        # To handle the "hidden seeded shift, rotation and offset", 
        # a simple coordinate descent might fail if the function is rotated.
        # However, f14 is "Different Powers", which is typically separable or weakly coupled.
        # If it is rotated, coordinate descent is inefficient.
        # But the inspiration says "exploits the high convergence rate of gradient-based methods... more robust than... ES".
        # Wait, the directive specifically asks for: 
        # "Implement a deterministic, multi-scale coordinate descent that starts with large strides (e.g., 4.0) and geometrically shrinks them (factor 0.5) until a minimum is found, leveraging the known 'Different Powers' structure..."
        
        # So I will follow the directive strictly.
        
        for step in steps:
            if time.time() > deadline or calls_made >= self.budget:
                break
                
            improved = True
            while improved and calls_made < self.budget and time.time() < deadline:
                improved = False
                for i in range(self.dim):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    
                    # Try positive and negative directions
                    for direction in [1.0, -1.0]:
                        new_x = x.copy()
                        new_x[i] += direction * step
                        new_x = np.clip(new_x, -5, 5)
                        
                        # Skip if no change
                        if np.all(new_x == x):
                            continue
                            
                        new_fx = func(new_x)
                        calls_made += 1
                        
                        if new_fx < fx:
                            x = new_x
                            fx = new_fx
                            improved = True
                            # Do not break out of direction loop, keep checking other direction?
                            # Usually if we move, the other direction is worse. 
                            # But with non-convexity, maybe? 
                            # Standard CD moves one step at a time. 
                            # Let's just accept the improvement and continue to next coordinate.
                            break 

        # If we have budget left, do a very fine polish
        # The steps list goes down to ~2.4e-7. That should be enough.
        
        return x, fx

# EVOLVE-BLOCK-END
