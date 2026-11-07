"""
Derivative-Free Coordinate Descent (DFCD) optimizer for Lunacek bi-Rastrigin (BBOB f24).
Uses a global grid scan followed by local coordinate descent refinement.
"""
import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        t0 = time.time()
        t_limit = 290.0
        n = self.dim
        budget = self.budget

        def elapsed():
            return time.time() - t0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            if self.calls >= budget or elapsed() > t_limit:
                return np.inf
            x = clip(x)
            f = func(x)
            self.calls += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # Phase 1: Global Coarse Grid Scan
        # 3 points per dimension in [-4, 4] -> 3^5 = 243 points
        grid_values = [-4.0, 0.0, 4.0]
        results = []
        
        for combo in itertools.product(grid_values, repeat=n):
            x = np.array(combo)
            f = evaluate(x)
            if f != np.inf:
                results.append((f, x))
            if self.calls >= budget or elapsed() > t_limit:
                break
        
        if not results:
            x = np.zeros(n)
            evaluate(x)
            return self.best_x, self.best_f

        # Sort results by fitness
        results.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement on the best 5 points
        top_k = min(5, len(results))
        candidates = [results[i][1].copy() for i in range(top_k)]
        
        for cand in candidates:
            if self.calls >= budget or elapsed() > t_limit:
                break
            
            current_x = cand.copy()
            current_f = evaluate(current_x)
            h = 2.0
            min_h = 1e-4
            
            # Run coordinate descent
            while h >= min_h and self.calls < budget and elapsed() < t_limit:
                improved_in_pass = False
                for d in range(n):
                    if self.calls >= budget or elapsed() > t_limit:
                        break
                    
                    # 1D search along dimension d
                    # Evaluate 4 neighbors: offsets -2, -1, 1, 2
                    best_f_local = current_f
                    best_x_local = current_x.copy()
                    
                    for offset in [-2, -1, 1, 2]:
                        p = current_x.copy()
                        p[d] += offset * h
                        f = evaluate(p)
                        if f < best_f_local:
                            best_f_local = f
                            best_x_local = p
                            
                    if best_f_local < current_f - 1e-10:
                        current_x = best_x_local
                        current_f = best_f_local
                        improved_in_pass = True
                    
                # Shrink step size
                h *= 0.5
                
        return self.best_x, self.best_f
