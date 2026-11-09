import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        call_count = 0
        def counted_func(x):
            nonlocal call_count
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            call_count += 1
            if call_count > self.budget:
                raise ValueError("Budget exceeded")
            return func(x)

        best_x = None
        best_f = np.inf
        
        # Strategy:
        # 1. Multi-start with Nelder-Mead
        # 2. Final polish with Powell or Nelder-Mead
        
        # We will dynamically allocate budget.
        # Let's reserve 50% for restarts, 50% for polish?
        # Or 80% restarts, 20% polish?
        # Rastrigin is multimodal, so finding the right basin is key.
        # 20 restarts with 20 calls each seems reasonable.
        
        n_restarts = 20
        fev_per_restart = 20
        
        # Ensure we don't exceed budget
        total_restart_fev = n_restarts * fev_per_restart
        if total_restart_fev > self.budget - 50: # Keep 50 for polish
            n_restarts = (self.budget - 50) // fev_per_restart
            if n_restarts < 1:
                n_restarts = 1
                fev_per_restart = self.budget - 50

        rng = np.random.default_rng(42)

        for i in range(n_restarts):
            if call_count >= self.budget - 10: # Stop if close to limit
                break
            if time.time() > deadline:
                break

            if i == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-5, 5, self.dim)
            
            try:
                res = minimize(
                    counted_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxfev': fev_per_restart, 'xatol': 1e-3, 'fatol': 1e-3}
                )
                
                # res.fun is the best value found in this local search
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, ValueError):
                break
            except Exception:
                # If minimize fails internally, try evaluating x0
                try:
                    f_val = counted_func(x0)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x0
                except:
                    pass

        # Final Polish
        if best_x is not None and call_count < self.budget - 5:
            remaining = self.budget - call_count
            if time.time() > deadline:
                return best_x, best_f
            
            try:
                # Use Powell for final precision
                res = minimize(
                    counted_func, 
                    best_x, 
                    method='Powell', 
                    options={'maxfev': remaining, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            try:
                best_f = counted_func(best_x)
            except:
                best_f = 0.0

        return best_x, best_f
