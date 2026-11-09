import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget

        calls_used = 0
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Global Search with (1+lambda)-ES
        # lambda = 4
        global_budget = 400
        local_budget = budget - global_budget
        
        # Initial point
        x = np.random.uniform(-5, 5, dim)
        fx = func(x)
        calls_used += 1
        best_x = x.copy()
        best_f = fx
        
        sigma = 1.0
        no_improve_count = 0
        lambda_pop = 4
        
        # (1+lambda)-ES loop
        while calls_used < global_budget and time.time() < deadline:
            # Generate lambda offspring
            offspring = []
            for _ in range(lambda_pop):
                z = np.random.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append((fy, y))
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
            # Select best offspring
            offspring.sort(key=lambda item: item[0])
            best_fy, best_y = offspring[0]
            
            # Success is if best offspring is better than parent
            if best_fy < fx:
                no_improve_count = 0
                x = best_y
                fx = best_fy
                # Adapt sigma: if success, increase slightly
                sigma *= 1.1
            else:
                no_improve_count += 1
                # Adapt sigma: if failure, decrease
                sigma /= 1.1
                
            # Stagnation check: if no improvement for many generations, restart
            # With lambda=4, we get 4 evaluations per generation.
            # 10 generations of no improvement = 40 calls.
            if no_improve_count > 10:
                # Restart from random point
                x = np.random.uniform(-5, 5, dim)
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve_count = 0
                sigma = 1.0 # Reset step size
                
            # Bound sigma
            sigma = np.clip(sigma, 0.01, 10.0)

        # Phase 2: Local Refinement with Nelder-Mead
        x0 = best_x.copy()
        
        nm_calls = 0
        nm_best_x = x0.copy()
        nm_best_f = best_f
        
        def wrapped_func(x):
            nonlocal nm_calls, nm_best_x, nm_best_f, calls_used
            if calls_used >= budget or time.time() > deadline:
                return 1e6
            x = np.clip(x, -5, 5)
            try:
                f_val = func(x)
            except:
                f_val = 1e6
            calls_used += 1
            nm_calls += 1
            if f_val < nm_best_f:
                nm_best_f = f_val
                nm_best_x = x.copy()
            return f_val
        
        def callback(xk):
            if calls_used >= budget or time.time() > deadline:
                return True
            return False
        
        try:
            simplex = [x0.copy()]
            for i in range(dim):
                p = x0.copy()
                p[i] += 0.1
                simplex.append(p)
                
            minimize(
                wrapped_func,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': local_budget,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True,
                    'initial_simplex': simplex
                },
                callback=callback
            )
        except:
            pass
        
        if nm_best_f < best_f:
            return nm_best_x, nm_best_f
        else:
            return best_x, best_f
