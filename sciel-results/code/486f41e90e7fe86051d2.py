"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Single-start (1+1)-ES with 1/5 success rule for rapid early convergence."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.perf_counter()
        deadline = start + 290.0  # Safety margin well under the 330 s kill
        rng = np.random.default_rng(0)
        budget = self.budget
        dim = self.dim

        # Start at the center of the box, which is a strong prior for BBOB instances
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        
        # Initial step size: large enough to explore, but not so large it wastes calls
        # On [-5, 5]^5, a step of 2.0 covers a significant portion of the space
        sigma = 2.0
        
        # 1/5 success rule parameters
        # We update the step size based on the success rate over the last 5 generations
        # If success rate > 0.2, increase sigma; if < 0.2, decrease it.
        # To keep it simple and fast, we use a running average or a window.
        # Standard (1+1)-ES often uses a simple multiplicative update per step.
        # Let's use the standard 1/5 rule with a window of 5.
        
        successes = []
        window_size = 5

        while calls < budget:
            if time.perf_counter() > deadline:
                break
                
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                successes.append(1)
            else:
                successes.append(0)
                
            if len(successes) >= window_size:
                success_rate = sum(successes) / window_size
                if success_rate > 0.2:
                    sigma *= 1.05 # Gentle increase
                elif success_rate < 0.2:
                    sigma *= 0.95 # Gentle decrease
                successes = [] # Reset window to keep responsiveness high

        return x, fx
# EVOLVE-BLOCK-END
