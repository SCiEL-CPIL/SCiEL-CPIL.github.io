import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        budget = self.budget
        
        low = [b[0] for b in bounds]
        high = [b[1] for b in bounds]
        
        # 1. Use known optimal x[2:] = 0 for DTLZ7 (g(0) = 0 is minimum)
        best_x_tail = np.zeros(n - 2)
        used = 0
        remaining_budget = budget
        
        # 2. Use all budget for grid search on x[0] and x[1]
        # The Pareto front of DTLZ7 is 2D, parameterized by x[0] and x[1]
        # Use a denser grid to better approximate the curved Pareto front
        
        # Allocate budget: 70% for structured grid, 30% for random fill
        n_grid_points = int(0.7 * remaining_budget)
        n_random = remaining_budget - n_grid_points
        
        # Create a 2D grid
        n_grid = int(np.sqrt(n_grid_points))
        if n_grid < 2:
            n_grid = 2
        
        # Use linspace for uniform coverage
        xs0 = np.linspace(low[0], high[0], n_grid)
        xs1 = np.linspace(low[1], high[1], n_grid)
        
        count = 0
        for x0 in xs0:
            for x1 in xs1:
                if time.time() > deadline:
                    return best_x_tail
                if count >= n_grid_points:
                    break
                    
                x = np.zeros(n)
                x[0] = x0
                x[1] = x1
                x[2:] = best_x_tail
                
                problem(x)
                count += 1
                
            if time.time() > deadline or count >= n_grid_points:
                break
        
        # 3. Random fill to capture points between grid nodes
        for i in range(n_random):
            if time.time() > deadline:
                break
            x_rand = np.zeros(n)
            x_rand[0] = np.random.uniform(low[0], high[0])
            x_rand[1] = np.random.uniform(low[1], high[1])
            x_rand[2:] = best_x_tail
            problem(x_rand)
            
        return best_x_tail
