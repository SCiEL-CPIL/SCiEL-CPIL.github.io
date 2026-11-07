"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Hybrid Global-Local Optimizer: Differential Evolution + Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        bounds = [(-5.0, 5.0)] * self.dim
        
        # Phase 1: Global Exploration with Differential Evolution
        # Use ~80% of budget for DE
        de_eval_budget = int(self.budget * 0.8)
        popsize = 10
        # Estimate maxiter to stay within budget
        # DE approx cost: popsize * (maxiter + 1)
        maxiter_de = max(1, de_eval_budget // popsize - 1)
        
        best_x = np.zeros(self.dim)
        best_f = np.inf
        
        try:
            res_de = differential_evolution(
                func, 
                bounds, 
                maxiter=maxiter_de, 
                popsize=popsize, 
                tol=1e-6, 
                seed=0, 
                polish=False, 
                init='sobol'
            )
            best_x = res_de.x
            best_f = res_de.fun
        except Exception:
            # Fallback if DE fails
            best_x = np.random.uniform(-5, 5, self.dim)
            best_f = func(best_x)

        # Phase 2: Local Exploitation with Nelder-Mead
        # Use remaining budget for NM
        # Check time
        if time.time() < deadline:
            try:
                # NM is local, so maxiter can be higher, but limited by time/budget
                # 50 iterations is usually enough for 5D
                res_nm = minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res_nm.fun < best_f:
                    best_x = res_nm.x
                    best_f = res_nm.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
