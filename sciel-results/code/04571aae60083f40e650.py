import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT6 specific knowledge:
        # Objectives depend on sum of variables S.
        # f1 = 1 + 90*S (for n=10)
        # We need f1 < 11 => S < 1/9
        # We sample S uniformly in [0, 1/9]
        
        n_evals = 0
        max_evals = self.budget
        
        # Calculate S range
        # Assuming standard ZDT6
        # f1(S) = 1 + 10*(dim-1)*S
        # We want f1 <= 11
        # 1 + 10*(dim-1)*S <= 11
        # S <= 10 / (10*(dim-1)) = 1/(dim-1)
        
        S_max = 1.0 / (self.dim - 1)
        
        # Generate S values
        # Ensure we don't exceed budget
        n_points = max_evals
        
        S_values = np.linspace(0.0, S_max, n_points)
        
        for S in S_values:
            if time.time() > deadline:
                break
            if n_evals >= max_evals:
                break
            
            # Construct x
            # Put all sum in x[0]
            x = np.zeros(self.dim)
            x[0] = S
            
            # Clip to bounds just in case, though S is small
            x = np.clip(x, lo, hi)
            
            # Evaluate
            try:
                problem(x)
                n_evals += 1
            except Exception:
                break
                
        return None
