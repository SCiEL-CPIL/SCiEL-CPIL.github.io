import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 330s limit
        
        # State tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f
            # Time check
            if time.time() - start_time > time_limit:
                return best_f
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(func, x0, max_iter, budget_remaining, time_remaining):
            """Nelder-Mead optimization with adaptive parameters."""
            dim = len(x0)
            # Initial simplex
            step_size = 0.5
            simplex = [x0.copy()]
            for i in range(dim):
                v = x0.copy()
                v[i] += step_size
                simplex.append(v)
            
            f_values = [func(simplex[0])]
            for i in range(1, dim + 1):
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
                f_values.append(func(simplex[i]))
            
            while len(f_values) < dim + 1:
                f_values.append(float('inf'))
            
            # Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
                
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                f_min = f_values[0]
                f_max = f_values[-1]
                if abs(f_max - f_min) < 1e-10 * max(1.0, abs(f_min)):
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = func(simplex[i])
            
            return simplex[0].copy(), f_values[0]

        # Phase 1: Multiple random restarts with Nelder-Mead
        # Rosenbrock has a narrow curved valley; Nelder-Mead handles this better than coordinate descent
        n_restarts = 15
        best_overall_x = None
        best_overall_f = float('inf')
        
        # Allocate budget for each restart
        budget_per_restart = self.budget // (n_restarts + 2)  # Leave some for final polish
        
        for restart_idx in range(n_restarts):
            if calls_made >= self.budget or time.time() - start_time > time_limit:
                break
            
            # Random starting point - sample from a wider range
            x = np.random.uniform(-4, 4, self.dim)
            fx = safe_eval(x)
            
            if fx >= best_overall_f:
                continue
            
            # Run Nelder-Mead from this starting point
            x_opt, f_opt = nelder_mead(safe_eval, x, max_iter=50, 
                                        budget_remaining=self.budget - calls_made,
                                        time_remaining=time_limit - (time.time() - start_time))
            
            if f_opt < best_overall_f:
                best_overall_f = f_opt
                best_overall_x = x_opt.copy()
        
        # Phase 2: Final polish with Nelder-Mead from the best result
        if best_overall_x is not None and calls_made < self.budget:
            x0 = best_overall_x.copy()
            x_opt, f_opt = nelder_mead(safe_eval, x0, max_iter=100,
                                        budget_remaining=self.budget - calls_made,
                                        time_remaining=time_limit - (time.time() - start_time))
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        return best_x, best_f
