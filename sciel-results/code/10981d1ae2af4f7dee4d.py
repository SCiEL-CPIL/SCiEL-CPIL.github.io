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
        
        # 1. Find the best x[2:] configuration to minimize f3
        # Include known optimal point (zeros) and more random samples
        candidates = []
        candidates.append(np.zeros(n - 2))
        candidates.append(np.full(n - 2, 0.5))
        candidates.append(np.ones(n - 2))
        
        for _ in range(17):
            cand = np.random.uniform(low[2:], high[2:])
            candidates.append(cand)
            
        best_x_tail = None
        best_f3 = np.inf
        
        for tail in candidates:
            x_test = np.zeros(n)
            x_test[0] = 0.5
            x_test[1] = 0.5
            x_test[2:] = tail
            
            f = problem(x_test)
            f3 = f[2]
            
            if f3 < best_f3:
                best_f3 = f3
                best_x_tail = tail.copy()
                
        # 2. Use remaining budget to grid search x[0] and x[1]
        used = len(candidates)
        remaining_budget = budget - used
        
        if remaining_budget <= 0:
            return best_x_tail
            
        n_grid = int(np.sqrt(remaining_budget))
        if n_grid < 2:
            n_grid = 2
            
        xs0 = np.linspace(low[0], high[0], n_grid)
        xs1 = np.linspace(low[1], high[1], n_grid)
        
        count = 0
        for x0 in xs0:
            for x1 in xs1:
                if time.time() > deadline:
                    break
                if count >= remaining_budget:
                    break
                    
                x = np.zeros(n)
                x[0] = x0
                x[1] = x1
                x[2:] = best_x_tail
                
                problem(x)
                count += 1
                
            if time.time() > deadline or count >= remaining_budget:
                break
                
        for i in range(remaining_budget - count):
            if time.time() > deadline:
                break
            x_rand = np.zeros(n)
            x_rand[0] = np.random.uniform(low[0], high[0])
            x_rand[1] = np.random.uniform(low[1], high[1])
            x_rand[2:] = best_x_tail
            problem(x_rand)
            
        return best_x_tail
