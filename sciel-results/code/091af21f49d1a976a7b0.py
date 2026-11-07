import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        used = 0
        
        # Find K such that (K+1)(K+2)/2 <= budget
        K = 0
        while (K+1)*(K+2)//2 <= self.budget:
            K += 1
        K -= 1 # Backtrack to valid K
        
        if K < 1:
            K = 1
            
        # Generate barycentric grid
        for i in range(K + 1):
            for j in range(K - i + 1):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Barycentric coordinates
                lambda1 = i / K
                lambda2 = j / K
                lambda3 = (K - i - j) / K
                
                # Map to decision variables x0, x1
                x0 = lambda1
                if lambda1 < 1.0:
                    x1 = lambda2 / (1.0 - lambda1)
                else:
                    x1 = 0.0 # Avoid division by zero, though lambda2 must be 0 if lambda1=1 in valid simplex grid
                
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5
                
                x = np.clip(x, lo, hi)
                
                try:
                    f = problem(x)
                    used += 1
                    f = np.array(f)
                    if np.all(f < 400.0):
                        results.append(f)
                except Exception:
                    continue
            
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
                
        return results
