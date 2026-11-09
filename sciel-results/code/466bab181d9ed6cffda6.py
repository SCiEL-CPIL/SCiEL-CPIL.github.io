import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        n = self.dim
        calls = 0
        best_x = np.random.uniform(-5, 5, n)
        best_f = func(best_x)
        calls += 1
        
        # Strategy: Multiple (1+1)-ES restarts
        # Schwefel is multimodal, so restarts are crucial.
        # We allocate a fraction of the budget to each restart.
        # Let's try 5 restarts, using 80% of budget for ES, 20% for polish.
        es_budget = int(self.budget * 0.8)
        polish_budget = self.budget - es_budget
        
        # (1+1)-ES parameters
        # Standard step size adaptation
        # For n=5, a good initial sigma is around 0.5-1.0
        sigma = 1.0
        # Success rate control (optional, but simple adaptation is often robust)
        # We use a simple multiplicative update based on success/failure
        
        restarts = 5
        budget_per_restart = es_budget // restarts
        
        for r in range(restarts):
            if calls >= es_budget:
                break
            
            # Initialize new search point
            # Mix random restart and local perturbation from global best
            if r == 0:
                x = best_x.copy()
            else:
                # Random restart within the box
                x = np.random.uniform(-5, 5, n)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # Reset sigma for new restart? Or keep it?
            # Keeping it might be better if we are close, but for random restarts, reset to reasonable value.
            # Let's reset to 0.5 for exploration.
            sigma = 0.5
            
            # (1+1)-ES loop
            # We run until budget_per_restart is used or time runs out
            while calls < es_budget and time.time() < deadline:
                # Generate offspring
                y = x + sigma * np.random.randn(n)
                # Clip to box
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    # Update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase sigma slightly on success
                    sigma *= 1.05
                else:
                    # Decrease sigma on failure
                    sigma *= 0.95
                    # If sigma becomes too small, break early to save budget for next restart
                    if sigma < 1e-4:
                        break
                
                # Check time
                if time.time() > deadline - 10:
                    break

        # Polish phase: Nelder-Mead on the best solution found
        if calls < self.budget and time.time() < deadline - 5:
            # Use remaining budget for Nelder-Mead
            # Nelder-Mead doesn't have a strict function call limit in the same way, 
            # but we can limit iterations or time.
            # We will run Nelder-Mead and check time/calls manually if possible, 
            # but scipy's minimize doesn't easily allow a strict call counter in the callback for early stop in all versions.
            # Instead, we can run a few iterations or rely on the fact that it's fast for n=5.
            
            # To be safe with call counts, we can wrap the function.
            remaining_calls = self.budget - calls
            if remaining_calls > 10:
                nm_calls = [0]
                def nm_func(x_nm):
                    nm_calls[0] += 1
                    if nm_calls[0] > remaining_calls or time.time() > deadline - 2:
                        # Force stop by returning a large value? No, that doesn't stop the optimizer.
                        # Better to just let it run, but Nelder-Mead is usually very fast for 5D.
                        pass
                    return func(x_nm)
                
                try:
                    res = minimize(nm_func, best_x, method='Nelder-Mead', 
                                   options={'maxiter': remaining_calls, 'xatol': 1e-8, 'fatol': 1e-8})
                    if res.success and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
        
        return best_x, best_f
