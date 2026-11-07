import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        total_calls = 0

        def time_ok():
            return time.time() < deadline

        def nelder_mead(func, x_start, max_calls):
            """
            Runs Nelder-Mead starting from x_start, using at most max_calls function evaluations.
            Returns (best_f, best_x, calls_used)
            """
            n = len(x_start)
            calls_used = 0
            
            # Initialize simplex
            simplex = [x_start.copy() for _ in range(n + 1)]
            step = 1.0
            for i in range(1, n + 1):
                simplex[i] = x_start.copy()
                simplex[i][i-1] += step
            
            # Evaluate initial simplex
            f_vals = []
            for i in range(n + 1):
                if calls_used >= max_calls or not time_ok():
                    break
                f_val = func(simplex[i])
                calls_used += 1
                f_vals.append(f_val)
                if f_val < best_f:
                    best_f = f_val
                    best_x = simplex[i].copy()
            
            if calls_used < n + 1:
                return best_f, best_x, calls_used

            # Main loop
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # We have n+1 calls used. Remaining is max_calls - calls_used.
            while calls_used < max_calls and time_ok():
                # Sort simplex
                idx = np.argsort(f_vals)
                simplex = [simplex[i] for i in idx]
                f_vals = [f_vals[i] for i in idx]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = func(x_ref)
                calls_used += 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
                
                if f_ref < f_vals[0]:
                    # Expansion
                    if calls_used >= max_calls or not time_ok():
                        break
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = func(x_exp)
                    calls_used += 1
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    # Contraction
                    if calls_used >= max_calls or not time_ok():
                        break
                    if f_ref < f_vals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    
                    f_con = func(x_con)
                    calls_used += 1
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    
                    if f_con < min(f_ref, f_vals[-1]):
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if calls_used >= max_calls or not time_ok():
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = func(simplex[i])
                            calls_used += 1
                            if f_vals[i] < best_f:
                                best_f = f_vals[i]
                                best_x = simplex[i].copy()
                        
                        if calls_used >= max_calls or not time_ok():
                            break

            return best_f, best_x, calls_used

        # Multi-start strategy
        # Use 10 starts. Allocate budget carefully.
        # 500 calls total. 10 starts.
        # NM needs at least 6 calls.
        # Let's give each start 45 calls to be safe (10 * 45 = 450).
        # This leaves 50 calls buffer for any overhead or if we add more starts later.
        # Or just 50 calls each, but ensure the loop breaks if budget is hit.
        
        np.random.seed(42)
        n_starts = 10
        start_points = [np.random.uniform(-4, 4, dim) for _ in range(n_starts)]
        start_points.append(np.zeros(dim)) # 11th point
        
        # We have 11 points. 500 / 11 is approx 45.
        # Let's iterate and allocate remaining budget, capped at 45 per start to prevent one from eating everything.
        for p in start_points:
            remaining = budget - total_calls
            if remaining < 6:
                break
            if not time_ok():
                break
            
            # Cap per-start calls to ensure we cover all starts
            chunk = min(45, remaining)
            
            local_best_f, local_best_x, local_calls = nelder_mead(func, p, chunk)
            
            best_f = local_best_f
            best_x = local_best_x
            total_calls += local_calls

        return best_x, best_f
