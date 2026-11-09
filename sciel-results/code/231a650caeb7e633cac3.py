# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Robust Evolution Strategy with Restarts for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Parameters for (mu+lambda) ES
        mu = 8
        lam = 8
        sigma = 3.0  # Initial step size
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Initial evaluation
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        calls_made += 1
        best_x = x0.copy()
        best_f = f0
        
        stall_count = 0
        
        while calls_made < budget and time.time() < deadline:
            # Generate new population
            offspring = []
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_child = best_x + sigma * z
                f_child = func(x_child)
                calls_made += 1
                offspring.append((f_child, x_child))
            
            if not offspring:
                break
            
            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Adapt sigma
            if stall_count == 0:
                sigma *= 0.95
            else:
                sigma *= 1.05
            
            # Restart logic
            if stall_count > 15:
                # If stuck, reduce sigma and perturb center
                sigma = max(0.5, sigma * 0.5)
                best_x = best_x + rng.uniform(-0.5, 0.5, dim)
                best_x = np.clip(best_x, -5, 5)
                stall_count = 0

            # If sigma becomes extremely small, force a global random restart
            if sigma < 1e-3:
                x_new = rng.uniform(-5, 5, dim)
                f_new = func(x_new)
                calls_made += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                sigma = 2.0
                stall_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
