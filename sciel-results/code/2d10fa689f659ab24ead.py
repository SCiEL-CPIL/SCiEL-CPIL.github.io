"""Adaptive (1+1)-ES for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-Evolution Strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Initialize at center
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        
        # Step size initialization
        sigma = 1.0
        
        # Success rate history for 1/5 rule
        successes = 0
        total = 0
        window = 10
        
        best_x = x.copy()
        best_fx = fx
        
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            
            # Generate perturbation
            z = np.random.randn(dim)
            y = x + sigma * z
            
            # Clip to box
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                successes += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Failure
                pass
                
            total += 1
            
            # Adapt step size using 1/5 success rule
            if total >= window:
                success_rate = successes / total
                if success_rate > 0.5:
                    sigma *= 1.5
                elif success_rate < 0.5:
                    sigma *= (1.0 / 1.5)
                successes = 0
                total = 0
            
            # Safety check for sigma
            if sigma < 1e-6:
                sigma = 1e-6
            elif sigma > 10.0:
                sigma = 10.0
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
