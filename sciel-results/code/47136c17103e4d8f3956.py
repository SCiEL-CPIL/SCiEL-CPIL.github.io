"""
Multi-start Nelder-Mead optimizer for BBOB Weierstrass f16.

Strategy:
1. Global exploration using Latin Hypercube Sampling (LHS) to cover the search space efficiently.
2. Select the best few candidates.
3. Refine each candidate using Nelder-Mead, which is robust to the non-smoothness of Weierstrass
   and has low overhead per function call compared to population-based methods like DE.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin for 300s limit
        
        n_calls = [0]
        stop_flag = [False]
        
        # Cache for function values to avoid redundant calls if Nelder-Mead re-evaluates start points
        # though Nelder-Mead usually doesn't, but LHS points might overlap if we were to do more.
        # For now, simple counter is enough.
        
        def safe_func(x):
            if stop_flag[0] or n_calls[0] >= budget or time.time() > deadline:
                stop_flag[0] = True
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Global Exploration with Latin Hypercube Sampling
        # 20 points is a good balance for 5D space to ensure coverage without wasting budget.
        n_global = 20
        if n_global >= budget:
            n_global = budget // 2 # Fallback if budget is very small
            
        # Generate LHS points
        sampler = LatinHypercube(d=dim, seed=42)
        lhs_samples = sampler.random(n=n_global)
        # Scale from [0, 1] to [-5, 5]
        lhs_samples = lhs_samples * 10.0 - 5.0
        
        # Evaluate global samples
        global_results = []
        for x in lhs_samples:
            if time.time() > deadline or n_calls[0] >= budget:
                break
            f_val = safe_func(x)
            if np.isfinite(f_val):
                global_results.append((f_val, x))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()

        if not global_results:
            return best_x, best_f

        # Sort by fitness
        global_results.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Select top K candidates
        # We want to balance the number of restarts with the budget per restart.
        # Nelder-Mead is fast. 100-200 calls is a good range for 5D.
        # Let's pick top 4.
        n_local = min(4, len(global_results))
        
        # Calculate remaining budget
        remaining_budget = budget - n_calls[0]
        
        if remaining_budget <= 10:
            return best_x, best_f
            
        # Allocate budget to local searches
        # We give slightly more budget to the better candidates?
        # Or just split evenly. Even split is simpler and robust.
        budget_per_local = int(remaining_budget / n_local)
        
        # Ensure at least some budget
        if budget_per_local < 10:
             # If budget is too small for multiple, just do one on the best
             n_local = 1
             budget_per_local = remaining_budget

        # Run Nelder-Mead on top candidates
        for i in range(n_local):
            if stop_flag[0] or n_calls[0] >= budget or time.time() > deadline:
                break
                
            f_start, x_start = global_results[i]
            
            # We don't need to re-evaluate x_start as it's already in global_results
            # But minimize might call it. We can pass a function that checks cache?
            # For simplicity, we just let it call. The overhead is 1 call.
            
            # Nelder-Mead options
            # maxfev: max function evaluations
            # xatol, fatol: tolerances
            # maxiter: max iterations
            
            # We set maxfev to the allocated budget.
            # We also set a time limit via the safe_func wrapper.
            
            try:
                res = minimize(
                    safe_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxfev': budget_per_local,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'maxiter': 1000,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        return best_x, best_f
