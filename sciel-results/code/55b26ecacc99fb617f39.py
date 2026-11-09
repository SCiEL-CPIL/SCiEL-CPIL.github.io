"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with stagnation-based restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        
        # Initialize with multiple diverse starting points
        num_starts = 10
        starts = rng.uniform(-4, 4, (num_starts, dim))
        
        best_x = None
        best_fx = float('inf')
        calls = 0

        # First pass: evaluate all starting points to find the best initial point
        for i in range(num_starts):
            if time.time() > deadline or calls >= budget:
                break
            x = starts[i]
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            best_x = x.copy()
            best_fx = fx

        # Run (1+1)-ES from the best starting point
        x = best_x.copy()
        fx = best_fx
        sigma = 2.0
        fail_count = 0
        stagnation_threshold = 30

        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Accept
                x, fx = y, fy
                sigma *= 1.3
                fail_count = 0
                
                # Update best
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Reject
                sigma *= 0.85
                fail_count += 1
                
                # Stagnation restart
                if fail_count >= stagnation_threshold:
                    # Try a few random restarts and pick the best one
                    restart_candidates = rng.uniform(-4, 4, (5, dim))
                    best_restart_x = None
                    best_restart_fx = float('inf')
                    
                    for rc in restart_candidates:
                        if calls >= budget or time.time() > deadline:
                            break
                        rc_fx = func(rc)
                        calls += 1
                        if rc_fx < best_restart_fx:
                            best_restart_fx = rc_fx
                            best_restart_x = rc.copy()
                    
                    if best_restart_x is not None:
                        x = best_restart_x
                        fx = best_restart_fx
                        sigma = 2.0
                        fail_count = 0
                        
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                    
                    if calls >= budget:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
