"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, NelderMead, L_BFGS_B

class Optimizer:
    """
    Hybrid Optimizer: Nelder-Mead for initial convergence, L-BFGS-B for polish.
    Designed for BBOB f12 (Bent Cigar) which is smooth, convex, and unimodal.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 300s limit, 20s safety margin
        n = self.dim
        budget = self.budget
        
        # State tracking
        x_best = np.zeros(n)
        f_best = np.inf
        calls_made = 0
        
        def eval_func(x):
            nonlocal calls_made, f_best, x_best
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if calls_made >= budget:
                raise ValueError("Budget exceeded")
            
            fx = float(func(x))
            calls_made += 1
            
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
                
            return fx

        # 1. Initial Phase: Nelder-Mead
        # Nelder-Mead is derivative-free and robust to the initial shape.
        # We give it a portion of the budget to find the basin.
        # For n=5, NM is very fast. Let's allocate 20% of the budget to NM.
        nm_budget = max(10, int(budget * 0.2))
        
        # Start at origin or random? Origin is safe for BBOB if shift is not huge, 
        # but random is safer against "hidden shift". 
        # However, BBOB shifts are usually within [-5,5]. 
        # Let's start at 0.0 as it's often close to the center of the box.
        x0 = np.zeros(n)
        
        # Run Nelder-Mead
        # We need to ensure NM doesn't blow up the budget.
        # scipy's NelderMead takes maxiter, maxfun.
        # We set maxfun to nm_budget.
        try:
            res_nm = minimize(
                eval_func, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': nm_budget, 
                    'maxfev': nm_budget, # maxfun in newer scipy
                    'xatol': 1e-8, 
                    'fatol': 1e-8
                }
            )
            if time.time() > deadline:
                return x_best, f_best
            
            x_current = res_nm.x
            f_current = res_nm.fun
            if f_current < f_best:
                f_best = f_current
                x_best = x_current.copy()
                
        except Exception as e:
            # If NM fails, fall back to current best
            pass

        # 2. Polish Phase: L-BFGS-B
        # L-BFGS-B uses gradients. Since func is black-box, we use finite differences.
        # This is much more efficient for smooth functions like Bent Cigar.
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x_best, f_best

        # L-BFGS-B parameters
        # We want to use as much of the remaining budget as possible for high precision.
        # maxfun controls the number of function evaluations.
        try:
            res_lbfgs = minimize(
                eval_func, 
                x_current, 
                method='L-BFGS-B', 
                bounds=[(-5.0, 5.0)] * n, # Explicit bounds
                options={
                    'maxiter': 1000, # Iterations, usually converges fast
                    'maxfev': remaining_budget, # Function evaluations
                    'gtol': 1e-12, # Gradient tolerance
                    'ftol': 1e-12, # Function tolerance
                    'disp': False
                }
            )
            
            if time.time() > deadline:
                return x_best, f_best
                
            if res_lbfgs.fun < f_best:
                f_best = res_lbfgs.fun
                x_best = res_lbfgs.x.copy()
                
        except Exception as e:
            # If L-BFGS-B fails, we keep the best from NM
            pass

        # 3. Final Polish: Nelder-Mead with tight tolerances
        # Sometimes L-BFGS-B gets stuck on non-smooth edges or numerical noise.
        # A final NM step can squeeze out a bit more if budget remains.
        remaining_budget = budget - calls_made
        if remaining_budget > 10:
            try:
                res_nm2 = minimize(
                    eval_func, 
                    x_best, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': remaining_budget, 
                        'maxfev': remaining_budget,
                        'xatol': 1e-12, 
                        'fatol': 1e-12
                    }
                )
                if res_nm2.fun < f_best:
                    f_best = res_nm2.fun
                    x_best = res_nm2.x.copy()
            except Exception:
                pass

        return x_best, f_best
# EVOLVE-BLOCK-END
