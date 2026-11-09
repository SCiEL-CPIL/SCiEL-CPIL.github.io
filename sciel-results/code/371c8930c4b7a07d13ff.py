"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        n = self.dim
        total_budget = self.budget
        
        best_x = None
        best_fx = np.inf
        rng = np.random.default_rng(42)
        
        # Track function calls
        call_count = 0
        calls_per_restart = total_budget // 10
        
        # Use 10 restarts with Nelder-Mead
        n_restarts = 10
        
        for r in range(n_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # Check budget
            if call_count >= total_budget:
                break
            
            # Initialize starting point
            x0 = rng.uniform(-4, 4, n)
            
            # Count calls for this starting point
            fx0 = func(x0)
            call_count += 1
            
            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()
            
            # Run Nelder-Mead from this starting point
            # Limit function evaluations
            remaining = total_budget - call_count
            maxfev = min(calls_per_restart, remaining)
            
            if maxfev < 10:
                break
            
            try:
                result = minimize(
                    func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': maxfev, 'maxfev': maxfev, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                
                # Count actual calls made
                call_count += result.nfev
                
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                # If Nelder-Mead fails, continue with next restart
                continue
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
