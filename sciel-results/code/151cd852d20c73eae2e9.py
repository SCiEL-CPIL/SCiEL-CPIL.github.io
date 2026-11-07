import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill
        
        rng = np.random.default_rng(42)
        
        # 1. Initial diverse sampling to find a good starting basin
        # Rosenbrock is narrow, so random uniform is often poor. 
        # We do a small batch of random starts, pick the best, then polish.
        
        n_initial = 20
        X = rng.uniform(-5, 5, (n_initial, self.dim))
        F = np.array([func(x) for x in X])
        
        best_idx = np.argmin(F)
        best_x = X[best_idx].copy()
        best_f = F[best_idx]
        
        calls_used = n_initial
        remaining = self.budget - calls_used
        
        # 2. Local Refinement Phase
        # Use L-BFGS-B for efficient convergence in the valley.
        # Rosenbrock is smooth, so gradient-based methods shine once in the basin.
        
        # We might need multiple local optimizations if the first one doesn't converge well
        # or if we can afford it. But L-BFGS-B is usually very efficient.
        
        # Let's try a robust local optimization from the best point.
        # We will use a simple loop of local optimizations with slightly perturbed starts 
        # if the first one doesn't improve enough, but given the budget, one good L-BFGS-B 
        # run from a decent start is often sufficient for 5D Rosenbrock.
        
        # However, to be safe and exploit the budget for AOCC (which rewards early progress),
        # we should ensure we get *some* improvement early. 
        # L-BFGS-B might take a few iterations to show significant improvement if the start is bad.
        # But we picked the best of 20 random starts, so it's likely in a reasonable region.
        
        # Let's run L-BFGS-B.
        # We need to track function calls inside scipy. 
        # Since we can't easily hook into scipy's internal call counter without wrapping,
        # we'll wrap the function to count calls.
        
        call_count = [0]
        
        def wrapper(x):
            if call_count[0] >= remaining:
                raise ValueError("Budget exceeded")
            call_count[0] += 1
            return func(x)
        
        try:
            res = minimize(wrapper, best_x, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'maxfun': remaining, 'maxls': 20})
            
            if res.success or res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            # If L-BFGS-B fails or exceeds budget, we stick with the best found so far.
            pass
        
        # 3. Optional: If we have budget left and time, do a quick Nelder-Mead polish
        # L-BFGS-B can sometimes get stuck in very flat regions or have numerical issues.
        # Nelder-Mead is derivative-free and robust.
        
        remaining = self.budget - (calls_used + call_count[0])
        
        if remaining > 50: # Enough for a quick NM run
            # Check time
            if time.time() > deadline - 10:
                return best_x, best_f
                
            # Reset call count for NM
            nm_call_count = [0]
            
            def nm_wrapper(x):
                if nm_call_count[0] >= remaining:
                    raise ValueError("Budget exceeded")
                nm_call_count[0] += 1
                return func(x)
                
            try:
                res_nm = minimize(nm_wrapper, best_x, method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
                    
            except Exception:
                pass
        
        return best_x, best_f
