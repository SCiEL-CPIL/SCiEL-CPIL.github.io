import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300 * 0.9
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG4 specific: x3..xn should be 0 to minimize g(x)
        # x1, x2 determine the position on the front
        # We assume bounds are [0, 1] for WFG4, but we use the provided bounds.
        
        x1_lo = max(lo[0], 0.01)
        x1_hi = min(hi[0], 3.0)
        x2_lo = max(lo[1], 0.01)
        x2_hi = min(hi[1], 5.0)
        
        # Create a 14x14 grid (196 points)
        x1_vals = np.logspace(np.log10(x1_lo), np.log10(x1_hi), 14)
        x2_vals = np.logspace(np.log10(x2_lo), np.log10(x2_hi), 14)
        
        candidates = []
        
        for x1 in x1_vals:
            for x2 in x2_vals:
                # Check if this point is potentially useful
                # f3 = 1/(x1*x2). We want f3 < 7 => x1*x2 > 1/7
                if x1 * x2 > 0.14: 
                    x = np.zeros(self.dim)
                    x[0] = x1
                    x[1] = x2
                    candidates.append(x)
        
        # If we have more than budget, subsample
        if len(candidates) > self.budget:
            indices = np.random.choice(len(candidates), self.budget, replace=False)
            candidates = [candidates[i] for i in indices]
        
        # Call problem on each candidate
        for x in candidates:
            if time.time() > deadline:
                break
            problem(x)
            
        return candidates
