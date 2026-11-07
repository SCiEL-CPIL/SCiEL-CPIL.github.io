import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
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
        # Use 55% of budget for global exploration
        global_budget = int(0.55 * self.budget)
        
        popsize = 10
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

        # Phase 2: Local Refinement with restarts
        # Strategy: Multiple rounds of local refinement with restarts
        # Each round: L-BFGS-B + (1+1)-ES, then check if we should restart
        remaining_budget = self.budget - calls
        if remaining_budget > 0 and best_x is not None and time_ok():
            rng = np.random.default_rng(123)
            num_restarts = 3  # Number of restart attempts
            
            for restart in range(num_restarts):
                if not time_ok() or calls >= self.budget:
                    break
                
                # For first restart, start from best found; for subsequent, perturb
                if restart == 0:
                    x = best_x.copy()
                else:
                    # Perturb from best with moderate noise to escape local minima
                    perturbation_scale = 1.0 * (restart + 1) / (num_restarts + 1)
                    x = best_x + perturbation_scale * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                
                fx = best_f if restart == 0 else wrapped_func(x)
                
                # Step 1: L-BFGS-B for fast gradient-based refinement
                if time_ok() and calls < self.budget:
                    try:
                        res_lbfgs = minimize(
                            wrapped_func,
                            x,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 30, 'ftol': 1e-15, 'gtol': 1e-10}
                        )
                        if res_lbfgs.fun < fx:
                            fx = res_lbfgs.fun
                            x = np.array(res_lbfgs.x, dtype=float).copy()
                    except (TimeoutError, Exception):
                        pass
                
                # Step 2: (1+1)-ES for fine-tuning
                if time_ok() and calls < self.budget:
                    sigma = 0.3  # Start moderate
                    local_rng = np.random.default_rng(123 + restart)
                    
                    # Use up to 30% of remaining budget for ES
                    es_budget = min(30, self.budget - calls)
                    
                    for _ in range(es_budget):
                        if not time_ok() or calls >= self.budget:
                            break
                        
                        y = x + sigma * local_rng.standard_normal(dim)
                        y = np.clip(y, -5, 5)
                        
                        fy = wrapped_func(y)
                        
                        if fy < fx:
                            x = y
                            fx = fy
                            sigma *= 1.15  # Increase on success
                        else:
                            sigma *= 0.85  # Decrease on failure
                        
                        if sigma < 1e-7:
                            break
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Check if we should continue with another restart
                # If we've made significant progress, continue; otherwise, stop
                if not time_ok() or calls >= self.budget:
                    break

        return best_x, best_f
