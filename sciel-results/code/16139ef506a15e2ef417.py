"""
Optimizer using coordinate-wise 1-D ternary search for de-shifting followed by
Nelder-Mead refinement for BBOB f20 (Schwefel) in 5 dimensions.
Structural change: decouple Schwefel's rugged landscape into tractable 1-D subproblems
by estimating local center per dimension, then refine jointly.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        calls = 0
        
        def check_budget():
            nonlocal calls
            return calls < budget and time.time() < deadline
        
        def eval_func(x):
            nonlocal calls
            if not check_budget():
                return float('inf')
            fx = func(x)
            calls += 1
            return fx
        
        # Phase 1: Coordinate-wise 1-D ternary search to estimate local centers
        # For each dimension, fix others at 0 and do ternary search in [-5, 5]
        center = np.zeros(n)
        
        for dim_idx in range(n):
            if not check_budget():
                break
            
            # Ternary search in 1-D
            a, b = -5.0, 5.0
            # Perform ~20 iterations of ternary search (uses ~40 evaluations)
            for _ in range(20):
                if not check_budget():
                    break
                m1 = a + (b - a) / 3.0
                m2 = b - (b - a) / 3.0
                
                x1 = center.copy()
                x1[dim_idx] = m1
                x2 = center.copy()
                x2[dim_idx] = m2
                
                f1 = eval_func(x1)
                f2 = eval_func(x2)
                
                if f1 < f2:
                    b = m2
                else:
                    a = m1
            
            # Estimate center as midpoint of final interval
            center[dim_idx] = (a + b) / 2.0
        
        # Evaluate the estimated center
        f_center = eval_func(center)
        best_x = center.copy()
        best_fx = f_center
        
        # Phase 2: Nelder-Mead refinement starting from the estimated center
        if not check_budget():
            return best_x, best_fx
        
        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        # Initialize Simplex around the estimated center
        step = 1.0
        simplex = np.zeros((n + 1, n))
        simplex[0] = center
        for i in range(n):
            simplex[i + 1] = center
            simplex[i + 1, i] += step
        
        # Evaluate initial simplex
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            if not check_budget():
                break
            fvals[i] = eval_func(simplex[i])
        
        if calls == 0:
            return best_x, best_fx
        
        # Main Nelder-Mead Loop
        while check_budget():
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
            
            if not check_budget():
                break
            f_ref = eval_func(x_ref)
            
            if f_ref < fvals[0]:
                # Reflection is better than best, try Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                
                if not check_budget():
                    simplex[worst_idx] = x_ref
                    fvals[worst_idx] = f_ref
                    break
                
                f_exp = eval_func(x_exp)
                
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
                
                if not check_budget():
                    break
                
                f_con = eval_func(x_con)
                
                if f_con < worst_f:
                    simplex[worst_idx] = x_con
                    fvals[worst_idx] = f_con
                else:
                    # Shrinkage
                    shrink_break = False
                    for i in range(1, n + 1):
                        if not check_budget():
                            shrink_break = True
                            break
                        
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = eval_func(simplex[i])
                    
                    if shrink_break:
                        break

        return best_x, best_fx
