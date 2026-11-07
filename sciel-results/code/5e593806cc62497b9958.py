"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES for BBOB-style f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(0)

        best_x = None
        best_fx = np.inf

        # Multi-restart strategy: split budget into several restarts
        # More restarts early to explore, fewer later to exploit
        n_restarts = 8
        # Allocate budget: decreasing allocation per restart
        # First restart gets the most, subsequent get less
        allocations = []
        remaining = budget
        for i in range(n_restarts):
            if i < n_restarts - 1:
                # Geometric decay
                alloc = max(30, int(remaining * 0.75))
                allocations.append(alloc)
                remaining -= alloc
            else:
                allocations.append(remaining)
        
        # Ensure we don't exceed budget
        total_alloc = sum(allocations)
        if total_alloc > budget:
            allocations[-1] = budget - sum(allocations[:-1])
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if allocations[r] <= 0:
                break

            # Initialize at a random point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            sigma = 1.0  # Initial step size
            
            # Run (1+1)-ES for this restart
            for step in range(allocations[r] - 1):
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.5
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87

        return best_x, best_fx
# EVOLVE-BLOCK-END
