"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with scouting and restart mechanism."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Scouting to find a good starting point
        # Evaluate origin and a few random points
        n_scout = 10
        if budget <= n_scout:
            # If budget is very small, just evaluate a few points and return best
            best_x = None
            best_f = float('inf')
            for i in range(budget):
                if i == 0:
                    x = np.zeros(dim)
                else:
                    x = np.random.uniform(-5, 5, dim)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x
            return best_x, best_f

        # Evaluate origin and random points
        best_x = np.zeros(dim)
        best_f = func(best_x)
        
        for i in range(n_scout - 1):
            x = np.random.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x

        # Phase 2: (1+1)-ES from the best scout
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        calls_made = n_scout
        no_improve_count = 0
        max_no_improve = 40  # Restart if stuck for 40 steps

        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * np.random.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x = y
                fx = fy
                no_improve_count = 0
                # Increase step size on success
                sigma *= 1.2
            else:
                no_improve_count += 1
                # Decrease step size on failure
                sigma *= 0.9
                
                # Restart mechanism: if stuck, increase step size to escape local minima
                if no_improve_count >= max_no_improve:
                    sigma = 2.0
                    no_improve_count = 0
                    # Optionally, we could jump to a new random point here, 
                    # but keeping the current best and just increasing step size is safer
                    # to maintain progress.
            
            # Keep sigma within bounds
            if sigma < 1e-4:
                sigma = 1e-4
            if sigma > 10.0:
                sigma = 10.0

        return x, fx
# EVOLVE-BLOCK-END
