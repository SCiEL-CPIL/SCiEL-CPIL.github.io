import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        # Phase 1: Global Exploration
        # Use 100 points to find the ridge
        n_explore = min(100, self.budget // 5) 
        points = rng.uniform(-5, 5, (n_explore, self.dim))
        values = np.array([func(p) for p in points])
        
        if time.time() > deadline:
            best_idx = np.argmin(values)
            return points[best_idx], values[best_idx]

        # Select top 5 points
        top_k = 5
        top_indices = np.argsort(values)[:top_k]
        top_points = points[top_indices]
        top_values = values[top_indices]
        
        # Phase 2: Local Exploitation with Parallel (1+1)-ES
        remaining_budget = self.budget - n_explore
        budget_per_es = remaining_budget // top_k
        
        best_x = top_points[0]
        best_f = top_values[0]
        
        for i in range(top_k):
            x = top_points[i].copy()
            fx = top_values[i]
            sigma = 2.0  # Start with a larger step to navigate to the ridge
            
            # Run (1+1)-ES for this instance
            for _ in range(budget_per_es):
                if time.time() > deadline:
                    break
                
                # Generate offspring
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2  # Moderate increase
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.7  # Moderate decrease
                    if sigma < 1e-6:
                        break
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
