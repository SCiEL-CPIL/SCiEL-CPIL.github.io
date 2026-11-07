import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Phase 1: Global Search using Latin Hypercube Sampling
        # Use 40% of the budget for global exploration
        n_global = max(10, min(200, int(self.budget * 0.4)))
        
        rng = np.random.default_rng(12345)
        
        # Generate LHS points
        points = np.zeros((n_global, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_global)
            u = rng.uniform(size=n_global)
            points[:, d] = (perm + u) * (10.0 / n_global) - 5.0
            
        # Evaluate all points
        results = []
        for i in range(n_global):
            x = points[i]
            f = func(x)
            results.append((f, x))
            
        # Sort by function value
        results.sort(key=lambda item: item[0])
        
        # Take top 2 candidates for local refinement
        # If we have less than 2, take what we have
        n_local_starts = 2 if len(results) >= 2 else len(results)
        
        # Distribute remaining budget among local starts
        remaining = self.budget - n_global
        budget_per_start = remaining // n_local_starts
        # If there's a remainder, give it to the first (best) start
        extra = remaining % n_local_starts
        
        best_x_final = None
        best_f_final = np.inf
        
        for i in range(n_local_starts):
            f_init, x_init = results[i]
            x = x_init.copy()
            fx = f_init
            
            sigma = 1.0
            # Use a different seed for each local run to decorrelate paths?
            # Or just continue with the same RNG.
            # Let's use a new RNG for each local run to be safe.
            local_rng = np.random.default_rng(1000 + i)
            
            steps = budget_per_start + (1 if i == 0 else 0)
            
            for _ in range(steps):
                y = x + sigma * local_rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87
            
            if fx < best_f_final:
                best_f_final = fx
                best_x_final = x

        return best_x_final, best_f_final
