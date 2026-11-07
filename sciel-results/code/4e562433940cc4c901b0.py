import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Use a fixed seed for reproducibility if needed, but random is fine for black-box
        # The task says "hidden seeded shift", implying the function is deterministic but unknown.
        # We can use a random seed or fixed seed. Fixed seed ensures same behavior on same instance.
        rng = np.random.default_rng(0)
        
        # Initialize at the center of the box
        x = np.zeros(self.dim)
        fx = func(x)
        
        # Initial step size. The domain is [-5, 5], so 2.0 is a reasonable start.
        sigma = 2.0
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        best_x = x.copy()
        best_f = fx
        
        # (1+1)-ES loop
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
                
            # Generate offspring
            # Standard normal noise
            z = rng.standard_normal(self.dim)
            y = x + sigma * z
            
            # Clip to box [-5, 5]
            y = np.clip(y, -5.0, 5.0)
            
            # Evaluate
            fy = func(y)
            
            # Update
            if fy < fx:
                x = y
                fx = fy
                best_x = x.copy()
                best_f = fx
                # Increase step size on success
                # Inspiration used 1.5, which is quite aggressive. 
                # For a quadratic, this might overshoot, but the next step will shrink.
                sigma *= 1.2 
            else:
                # Decrease step size on failure
                sigma *= 0.8
                
            # Optional: Prevent sigma from becoming too small or too large
            if sigma < 1e-10:
                break
            if sigma > 10.0:
                sigma = 10.0

        return best_x, best_f
