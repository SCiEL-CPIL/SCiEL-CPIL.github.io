"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with step-size adaptation and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        # Phase 1: (1+1)-ES with better sigma adaptation
        for i in range(self.budget - 50):
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            
            if fy < fx:
                step_norm = np.linalg.norm(y - x)
                if step_norm > 1e-10:
                    # Adapt sigma based on step acceptance and magnitude
                    target_step = 0.5 * np.sqrt(self.dim)
                    sigma *= 1.0 + 0.3 * np.log(max(target_step / max(step_norm, 1e-10), 1e-10))
                    sigma = np.clip(sigma, 1e-6, 10.0)
                x, fx = y, fy
            else:
                sigma *= 0.7
                sigma = max(sigma, 1e-8)
        
        # Phase 2: Nelder-Mead polish with remaining budget
        remaining_calls = self.budget - (i + 2)
        if remaining_calls > 10 and time.time() < deadline - 5.0:
            def wrapper(x_vec):
                return func(np.clip(x_vec, -5, 5))
            
            # Use a subset of remaining budget for NM
            nm_maxiter = min(remaining_calls, 100)
            try:
                res = minimize(wrapper, x, method='Nelder-Mead', 
                              options={'maxiter': nm_maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < fx:
                    x, fx = res.x, res.fun
            except Exception:
                pass
        
        return x, fx
# EVOLVE-BLOCK-END
