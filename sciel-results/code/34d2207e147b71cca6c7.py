"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with time-aware step control and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 290  # safety margin before 330s limit
        rng = np.random.default_rng(0)
        
        best_x = None
        best_fx = float('inf')
        
        # Multiple restarts to escape local optima (though Sphere is unimodal,
        # the shift/rotation/offset might make it tricky if we don't start close)
        # Actually Sphere is convex, so one run should be fine if we converge.
        # But let's use a more robust approach: CMA-ES like behavior or just better ES.
        
        # Let's try a (1+1)-ES with adaptive step size but also check if we should restart.
        # Since it's Sphere, global optimum is unique. The main risk is getting stuck
        # due to numerical issues or if the initial point is far and step size decays too fast.
        
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        
        sigma = 1.0
        calls = 1
        
        for i in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                sigma *= 1.2  # Increase step size on success
            else:
                sigma *= 0.85  # Decrease step size on failure
            
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # If we've made very little progress for a long time, consider a restart
            # But for Sphere, this shouldn't be necessary if sigma is well-tuned.
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
