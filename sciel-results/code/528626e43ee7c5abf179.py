"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 310
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_eval = 0
        all_points = []
        
        # Use global np.random to respect the seed set by the evaluator
        while n_eval < self.budget:
            if time.time() - start_time > DEADLINE:
                break
            
            # Generate a batch of random points
            batch_size = min(50, self.budget - n_eval)
            X_batch = np.random.uniform(lo, hi, size=(batch_size, self.dim))
            
            for x in X_batch:
                f = problem(x)
                all_points.append(f)
                n_eval += 1
                
                if time.time() - start_time > DEADLINE:
                    break
            
        return all_points
# EVOLVE-BLOCK-END
