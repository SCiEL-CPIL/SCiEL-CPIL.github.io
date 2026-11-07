"""Optimizer for BBOB-style f15 (Rastrigin rotated) with Multi-start (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Strategy: Single sustained (1+1)-ES with aggressive sigma adaptation
        # This allows deeper convergence in the main basin, which is crucial for Rastrigin
        # after finding the correct local minimum.
        
        # Initialize at origin or a random point? 
        # The previous multi-start was fragmented. Let's start with a single strong lineage.
        # However, Rastrigin has many local minima. 
        # The "knowledge" suggests: "Replacing fragmented multi-start with a single sustained (1+1)-ES... should allow deeper convergence per lineage and better early progress tracking"
        
        # Let's start at the origin (0,0,0,0,0) as it's a common attractor or close to one in rotated Rastrigin?
        # Actually, random start might be better to avoid getting stuck if origin is far.
        # But with aggressive sigma, it can move fast.
        
        # Let's try starting at 0.
        x = np.zeros(dim)
        fx = func(x)
        calls_made += 1
        best_f = fx
        best_x = x.copy()
        
        sigma = 2.0
        
        # Main loop
        while calls_made < budget and time.time() < deadline:
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87
                
            # Lightweight restart logic:
            # If we get stuck (sigma becomes very small) and we haven't found a very good solution,
            # we might restart from a random point.
            # But for now, let's stick to the single lineage to maximize depth.
            # If sigma drops below a threshold, we can increase it slightly to escape local minima?
            # The 1/5 rule handles this.
            
            # Check if we should break early if time is critical
            if time.time() > deadline - 5:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
