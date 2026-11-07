"""
Optimizer using Nelder-Mead Simplex Method for BBOB f20 (Schwefel) in 5 dimensions.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        # Initialize Simplex
        # Start near center, but spread out to cover the box [-5, 5]
        step = 2.5
        simplex = np.zeros((n + 1, n))
        for i in range(n):
            simplex[i + 1, i] = step
        
        # Evaluate initial simplex
        fvals = np.zeros(n + 1)
        calls = 0
        
        for i in range(n + 1):
            if calls >= budget or time.time() > deadline:
                break
            x = simplex[i]
            fx = func(x)
            fvals[i] = fx
            calls += 1
        
        if calls == 0:
            return np.zeros(n), 0.0

        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_fx = fvals[best_idx]
        
        # Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by fitness (best first)
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = simplex[0].copy()
                
            # Identify worst point
            worst_idx = n
            worst_x = simplex[worst_idx]
            worst_f = fvals[worst_idx]
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:n], axis=0)
            
            # 1. Reflection
            x_ref = centroid + alpha * (centroid - worst_x)
            x_ref = np.clip(x_ref, -5, 5)
            
            if calls >= budget or time.time() > deadline:
                break
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < fvals[0]:
                # Reflection is better than best, try Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                
                if calls >= budget or time.time() > deadline:
                    # If we can't expand, just accept reflection
                    simplex[worst_idx] = x_ref
                    fvals[worst_idx] = f_ref
                    break
                
                f_exp = func(x_exp)
                calls += 1
                
                if f_exp < f_ref:
                    simplex[worst_idx] = x_exp
                    fvals[worst_idx] = f_exp
                else:
                    simplex[worst_idx] = x_ref
                    fvals[worst_idx] = f_ref
                    
            elif f_ref < fvals[n - 1]:
                # Reflection is better than worst, but worse than second best
                simplex[worst_idx] = x_ref
                fvals[worst_idx] = f_ref
                
            else:
                # Reflection is worse than worst, try Contraction
                x_con = centroid + rho * (worst_x - centroid)
                x_con = np.clip(x_con, -5, 5)
                
                if calls >= budget or time.time() > deadline:
                    break
                
                f_con = func(x_con)
                calls += 1
                
                if f_con < worst_f:
                    simplex[worst_idx] = x_con
                    fvals[worst_idx] = f_con
                else:
                    # Shrinkage
                    # Shrink all points except the best towards the best
                    shrink_break = False
                    for i in range(1, n + 1):
                        if calls >= budget or time.time() > deadline:
                            shrink_break = True
                            break
                        
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                    
                    if shrink_break:
                        break

        return best_x, best_fx
