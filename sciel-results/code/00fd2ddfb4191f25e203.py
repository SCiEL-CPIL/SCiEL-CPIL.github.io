"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule and stall detection."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        # Start at the center, which is a strong prior for BBOB functions
        x = np.zeros(self.dim)
        fx = func(x)
        
        # Initial step size
        sigma = 1.0
        
        # 1/5 rule parameters
        window_size = 5
        successes = 0
        steps_in_window = 0
        
        # Stall detection
        steps_without_improvement = 0
        best_fx = fx
        stall_threshold = 30
        
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        for _ in range(self.budget - 1):
            # Time guard
            if time.time() > deadline:
                break
                
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                successes += 1
                if fx < best_fx:
                    best_fx = fx
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
            else:
                steps_without_improvement += 1
                
            steps_in_window += 1
            
            # Adapt step size based on 1/5 rule
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                if success_rate > 1/5:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                successes = 0
                steps_in_window = 0
                
            # Stall detection: if no improvement for a while, explore more
            if steps_without_improvement > stall_threshold:
                sigma *= 2.0
                steps_without_improvement = 0
                # Optionally, we could also try a random jump, but increasing sigma is safer
                # to avoid losing the current best position if it's close.
                
        return x, fx
# EVOLVE-BLOCK-END
