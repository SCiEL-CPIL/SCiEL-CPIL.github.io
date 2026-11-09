"""Rewritten program: Deterministic Global Grid + Local (1+5)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer:
    1. Coarse deterministic grid scan to find the global basin.
    2. Local (1+5)-ES to refine the minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 300.0 # Safety margin for 330s limit
        
        # Phase 1: Coarse Grid Scan
        # 3 points per dimension: -5, 0, 5
        grid_vals = np.array([-5.0, 0.0, 5.0])
        grids = np.meshgrid(*([grid_vals] * self.dim))
        grid_points = np.array([g.ravel() for g in grids]).T
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        for i in range(len(grid_points)):
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x = grid_points[i]
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return np.zeros(self.dim), float('inf')

        # Phase 2: Local (1+5)-ES
        remaining_budget = self.budget - calls_made
        
        if remaining_budget <= 0:
            return best_x, best_f

        x = best_x
        fx = best_f
        sigma = 1.0 # Initial step size
        lambda_ = 5
        
        while calls_made < self.budget and time.time() < deadline:
            # Generate lambda offspring
            offspring = []
            for j in range(lambda_):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * np.random.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))
            
            if not offspring:
                break
                
            # Select the best offspring
            offspring.sort(key=lambda item: item[0])
            best_fy, best_y = offspring[0]
            
            if best_fy < fx:
                x, fx = best_y, best_fy
                # Increase sigma if we improved
                sigma *= 1.1
            else:
                # Decrease sigma
                sigma *= 0.9
            
            # Keep sigma within bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
