"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5 using DE and NM."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Rastrigin-aware optimizer: Differential Evolution + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        call_count = [0]
        best_x = np.zeros(dim)
        best_f = np.inf

        def do_call(x):
            nonlocal best_f, best_x
            if call_count[0] >= budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            call_count[0] += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        bounds = [(-5.0, 5.0)] * dim
        
        # Phase 1: Differential Evolution for global search
        # DE is robust for multimodal functions like Rastrigin
        # Use Sobol initialization for better space coverage
        de_budget = int(budget * 0.8)
        
        # We need to wrap the function to count calls and enforce deadline
        def de_func(x):
            if not time_ok() or call_count[0] >= de_budget:
                return best_f
            return do_call(x)

        try:
            # popsize=10 is a good balance for 5D with limited budget
            # tol/xtol set to stop when converged
            res = differential_evolution(
                de_func, 
                bounds, 
                maxiter=100, 
                maxfev=de_budget,
                popsize=10,
                init='sobol',
                seed=42,
                tol=1e-10,
                xtol=1e-10,
                polish=False # We will do our own polish
            )
        except Exception:
            pass

        # Update best if DE found something better
        if res.success or hasattr(res, 'x'):
            if call_count[0] < budget:
                val = do_call(res.x)
        
        # Phase 2: Nelder-Mead Polish on the best candidate
        remaining_budget = budget - call_count[0]
        if remaining_budget > 10 and time_ok():
            nm_calls = [0]
            
            def nm_func(x, nm_calls=nm_calls):
                if nm_calls[0] >= remaining_budget or call_count[0] >= budget or not time_ok():
                    return best_f
                nm_calls[0] += 1
                return do_call(x)
            
            try:
                # Use best_x from DE as start point
                res_nm = minimize(nm_func, best_x, method='Nelder-Mead',
                               options={'maxiter': remaining_budget, 'xatol': 1e-12, 
                                        'fatol': 1e-12, 'adaptive': True, 
                                        'maxfev': remaining_budget})
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
