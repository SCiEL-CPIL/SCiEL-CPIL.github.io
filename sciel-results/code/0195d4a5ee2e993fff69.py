"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead Simplex Optimizer with Coordinate Descent Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, max_iter):
            nonlocal calls_used, deadline, best_x, best_f
            
            # Initialize simplex
            simplex = []
            f_simplex = []
            
            # First point
            x0 = np.clip(x0, -5, 5)
            f0 = call_func(x0)
            simplex.append(x0.copy())
            f_simplex.append(f0)
            
            # Generate other vertices
            step = 0.5
            for i in range(dim):
                x_new = x0.copy()
                x_new[i] += step
                x_new = np.clip(x_new, -5, 5)
                f_new = call_func(x_new)
                simplex.append(x_new.copy())
                f_simplex.append(f_new)
                
            simplex = np.array(simplex)
            f_simplex = np.array(f_simplex)
            
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrinkage
            
            for _ in range(max_iter):
                if time.time() > deadline or calls_used >= budget:
                    break
                
                # Sort by function value
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]
                
                # Check convergence
                if np.std(f_simplex) < 1e-12:
                    break
                
                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = call_func(x_ref)
                
                if f_ref < f_simplex[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = call_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_simplex[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = call_func(x_con)
                    
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = call_func(simplex[i])
                        
            return simplex[0], f_simplex[0]

        def coordinate_descent(x0, max_iter):
            nonlocal calls_used, deadline, best_x, best_f
            
            x = x0.copy()
            f_x = call_func(x)
            step = 0.5
            
            for _ in range(max_iter):
                if time.time() > deadline or calls_used >= budget:
                    break
                
                improved = False
                for i in range(dim):
                    if time.time() > deadline or calls_used >= budget:
                        break
                    
                    # Try positive step
                    x_trial = x.copy()
                    x_trial[i] += step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = call_func(x_trial)
                    
                    if f_trial < f_x:
                        x = x_trial
                        f_x = f_trial
                        improved = True
                    else:
                        # Try negative step
                        x_trial = x.copy()
                        x_trial[i] -= step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = call_func(x_trial)
                        
                        if f_trial < f_x:
                            x = x_trial
                            f_x = f_trial
                            improved = True
                        else:
                            # No improvement, reduce step size for this coordinate
                            pass
                
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
                else:
                    # If improved, keep step or increase slightly
                    step = min(step * 1.2, 2.0)
                    
            return x, f_x

        # Strategy:
        # 1. Generate diverse initial points
        # 2. Run Nelder-Mead from the best few starts
        # 3. Polish with coordinate descent
        
        rng = np.random.default_rng(42)
        
        starts = []
        # Standard origin
        starts.append(np.zeros(dim))
        
        # Theoretical minimum location
        starts.append(np.ones(dim))
        
        # Random points
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, dim))
            
        # Diagonal points
        for val in [-2, 0, 2]:
            starts.append(np.full(dim, val))
        
        # Evaluate all starts
        best_start_x = None
        best_start_f = np.inf
        start_results = []
        
        for s in starts:
            if time.time() > deadline or calls_used >= budget:
                break
            s = np.clip(s, -5, 5)
            f_s = call_func(s)
            start_results.append((s, f_s))
            if f_s < best_start_f:
                best_start_f = f_s
                best_start_x = s.copy()
        
        # Sort starts by function value
        start_results.sort(key=lambda x: x[1])
        
        # Run Nelder-Mead from top 3 starts
        # Allocate budget dynamically
        remaining_calls = budget - calls_used
        
        if best_start_x is not None:
            # Main run from best start
            max_iter_main = int(remaining_calls * 0.4)
            if max_iter_main > 10:
                x_main, f_main = nelder_mead(best_start_x, max_iter_main)
                
                # Polish with coordinate descent
                remaining_calls = budget - calls_used
                if remaining_calls > 20:
                    max_iter_cd = int(remaining_calls * 0.3)
                    if max_iter_cd > 5:
                        x_cd, f_cd = coordinate_descent(x_main, max_iter_cd)
                        
                        # If we have more budget, try second best start
                        remaining_calls = budget - calls_used
                        if remaining_calls > 50 and len(start_results) > 1:
                            second_best_x = start_results[1][0]
                            max_iter_second = int(remaining_calls * 0.5)
                            if max_iter_second > 10:
                                x_second, f_second = nelder_mead(second_best_x, max_iter_second)
                                
                                # Polish second result
                                remaining_calls = budget - calls_used
                                if remaining_calls > 10:
                                    max_iter_cd2 = int(remaining_calls * 0.8)
                                    if max_iter_cd2 > 5:
                                        x_cd2, f_cd2 = coordinate_descent(x_second, max_iter_cd2)

        return best_x, best_f
# EVOLVE-BLOCK-END
