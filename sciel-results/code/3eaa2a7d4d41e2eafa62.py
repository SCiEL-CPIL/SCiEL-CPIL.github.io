"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 300.0
        
        # Parameters
        # For (1+1)-ES, we generate 1 offspring per generation.
        # 1/5 success rule: increase step size if >20% of steps succeed.
        # For (1+1), we can use a sliding window or just a simple counter.
        
        # Initial state
        # Lunacek bi-Rastrigin has two minima. One is global, one is local.
        # The global minimum is typically at x* = -1.5 * 1^d (or similar shift).
        # The local minimum is at x = 0.
        # Starting at 0 is risky if the global minimum is far away.
        # Starting at a random point is better.
        # However, for BBOB, the shift is usually small or structured.
        # Let's start at 0.0, as it's the center of the domain and often close to a minimum.
        # If we get stuck, the 1/5 rule will expand the step size.
        
        rng = np.random.default_rng(0)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Step size
        # Initial step size should be large enough to explore the domain.
        # Domain is [-5, 5]^d.
        # sigma = 2.0 is a reasonable start.
        sigma = 2.0
        
        # 1/5 success rule parameters
        # We need a window size. Common choice is 1/d or 1.
        # Let's use a window of size 10 or so? 
        # Actually, the standard 1/5 rule uses a window of size N = 1/d * (some factor).
        # For simplicity, let's use a fixed window size of 10.
        # If more than 2 successes in 10 trials, increase sigma.
        # If less than 2, decrease sigma.
        
        window_size = 10
        successes = 0
        trials = 0
        
        # Alternative: use a simple adaptive rule without window.
        # If success: sigma *= 1.1
        # If failure: sigma /= 1.1
        # This is equivalent to a very short window.
        # The 1/5 rule is more robust.
        
        # Let's implement the 1/5 rule properly.
        # We need to store the last N steps.
        
        while calls < budget:
            if time.time() - start_time > max_time:
                break
            
            # Generate offspring
            # y = x + sigma * N(0, I)
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            
            # Success rule
            # Success if fy < fx (improvement)
            if fy < fx:
                successes += 1
                # Accept step
                x = y.copy()
                fx = fy
            else:
                # Reject step
                pass
            
            trials += 1
            
            # Update sigma if window is full
            if trials >= window_size:
                # Check success rate
                success_rate = successes / trials
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma /= 1.1
                
                # Reset window
                successes = 0
                trials = 0
        
        return best_x, best_f

# EVOLVE-BLOCK-END
