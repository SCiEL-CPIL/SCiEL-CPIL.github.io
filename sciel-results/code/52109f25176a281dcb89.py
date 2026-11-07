import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf
        dim = self.dim

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                raise TimeoutError("Time limit exceeded")
            calls += 1
            if calls > self.budget:
                raise Exception("Budget exceeded")
            
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # Phase 1: Global Search with Differential Evolution
        # Use 60% of budget for global exploration to ensure we find the basin
        global_budget = int(0.60 * self.budget)
        
        popsize = 10
        # DE uses popsize for init + popsize per generation
        # Total calls approx popsize * (1 + max_iter)
        # We want this <= global_budget
        max_iter_de = max(1, int(global_budget / popsize) - 1)
        
        try:
            res_de = differential_evolution(
                wrapped_func, 
                bounds=[(-5, 5)] * dim, 
                maxiter=max_iter_de,
                popsize=popsize,
                polish=False,
                tol=1e-10,
                seed=42,
                init='latinhypercube',
                strategy='best1bin'
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = np.array(res_de.x, dtype=float).copy()
        except (TimeoutError, Exception):
            pass

        # Phase 2: Local Refinement with (1+1)-ES and Restarts
        # The (1+1)-ES is very efficient (1 call per step) and good for fine-tuning
        # We use a multi-restart strategy to avoid getting stuck in local minima
        remaining_budget = self.budget - calls
        
        if remaining_budget > 0 and best_x is not None and time_ok():
            rng = np.random.default_rng(123)
            
            # Split remaining budget into 2 phases of local search
            # Phase 2a: Refine from best DE point
            # Phase 2b: Restart with perturbation if needed
            
            phase_budget = int(0.7 * remaining_budget)
            if phase_budget < 10:
                phase_budget = remaining_budget
            
            # --- Local Search 1 ---
            if time_ok() and calls < self.budget:
                x = best_x.copy()
                fx = best_f
                sigma = 0.5
                local_rng = np.random.default_rng(1)
                
                # Run (1+1)-ES for phase_budget steps
                for _ in range(phase_budget):
                    if not time_ok() or calls >= self.budget:
                        break
                    
                    y = x + sigma * local_rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    
                    fy = wrapped_func(y)
                    
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.85
                        
                    if sigma < 1e-8:
                        break
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # --- Local Search 2 (Restart) ---
            remaining_budget = self.budget - calls
            if remaining_budget > 10 and best_x is not None and time_ok():
                # Perturb the best point to escape potential local minima
                # Use a moderate perturbation
                perturbation_scale = 0.5
                x = best_x + perturbation_scale * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                
                # Evaluate the perturbed point to get starting value
                # If it's worse, we might want to start from best_x anyway, 
                # but (1+1)-ES will handle the move back if needed.
                # However, to save calls, we can just start the loop.
                # But we need a valid fx.
                try:
                    fx = wrapped_func(x)
                except:
                    x = best_x
                    fx = best_f

                sigma = 0.3
                local_rng2 = np.random.default_rng(2)
                
                # Use remaining budget
                for _ in range(self.budget - calls):
                    if not time_ok() or calls >= self.budget:
                        break
                    
                    y = x + sigma * local_rng2.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    
                    fy = wrapped_func(y)
                    
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.85
                        
                    if sigma < 1e-8:
                        break
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
