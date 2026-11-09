"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State variables
        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0
        
        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return best_f
            call_count += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except:
                return 1e100

        bounds = [(-5, 5)] * dim
        
        # Strategy: 
        # 1. Use Differential Evolution (DE) for robust global search to locate the valley.
        #    DE is population-based and handles the rotated/shifted landscape well without 
        #    relying on local gradients that might mislead in the narrow valley.
        # 2. Use L-BFGS-B for a high-precision polish on the best candidate found by DE.
        #    L-BFGS-B is efficient for smooth functions once in the basin of attraction.
        
        # Phase 1: Differential Evolution
        # We allocate roughly 70% of the budget to DE.
        # DE parameters:
        # - popsize: 15 (default is 15*dim=75, which is too high. 15 is reasonable for dim=5)
        # - maxiter: We estimate iterations based on budget.
        #   Each iteration uses popsize evaluations.
        #   Budget for DE = 350. Popsize = 15. Max iter ~ 20.
        #   However, DE convergence can be slow. Let's set maxiter high and let it stop if no improvement or budget runs out.
        #   Actually, scipy's DE has a 'maxiter' option. 
        #   Let's use a moderate popsize and let it run.
        
        # We need to be careful with the budget.
        # scipy.optimize.differential_evolution does not have a strict call limit callback that stops it immediately 
        # inside the loop if we exceed budget, but our counted_func returns best_f (a constant) 
        # once budget is exceeded. This might cause DE to think it converged or behave oddly.
        # Better approach: Estimate maxiter such that popsize * maxiter <= budget * 0.7.
        
        de_popsize = 12
        de_max_iter = int((budget * 0.7) / de_popsize)
        
        # Ensure at least 1 iteration
        de_max_iter = max(1, de_max_iter)
        
        try:
            # Use a fixed seed for reproducibility if needed, but random is fine.
            # DE is stochastic.
            de_result = differential_evolution(
                counted_func, 
                bounds=bounds, 
                popsize=de_popsize,
                maxiter=de_max_iter,
                tol=1e-6,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                polish=False  # We will do our own polish with L-BFGS-B
            )
            
            if de_result.fun < best_f:
                best_f = de_result.fun
                best_x = de_result.x.copy()
                
        except Exception as e:
            pass
            
        # Phase 2: L-BFGS-B Polish
        # Use the remaining budget for a precise local optimization.
        if call_count < budget - 10 and time.time() < deadline:
            try:
                # L-BFGS-B is very efficient for smooth functions like Rosenbrock.
                # It uses gradient information (via finite differences).
                # With dim=5, this is cheap.
                res = minimize(
                    counted_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception as e:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
