"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with success rate control and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Initialization
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 1.0
        
        # Success rate control parameters
        window_size = 10
        successes = 0
        success_history = []
        
        # Main (1+1)-ES loop
        # We reserve some budget for polishing
        polish_budget = 50
        es_budget = self.budget - polish_budget
        
        for i in range(es_budget):
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            
            # Update success history
            if fy < fx:
                x, fx = y, fy
                success_history.append(1)
            else:
                success_history.append(0)
            
            # Maintain window
            if len(success_history) > window_size:
                success_history.pop(0)
            
            # Adapt sigma based on success rate
            if len(success_history) == window_size:
                success_rate = np.mean(success_history)
                if success_rate > 0.2:
                    sigma *= 1.05
                elif success_rate < 0.2:
                    sigma *= 0.95
        
        # Polish with Nelder-Mead if budget and time allow
        if time.time() < deadline - 10.0: # Leave 10s margin for polish
            # Nelder-Mead doesn't have a strict call limit in the same way, 
            # but we can limit iterations. 
            # We have roughly polish_budget calls left.
            # Nelder-Mead uses (dim+1) points per iteration.
            # Max iterations = polish_budget // (dim + 1)
            max_iter = polish_budget // (self.dim + 1)
            
            try:
                res = minimize(
                    func, 
                    x, 
                    method='Nelder-Mead', 
                    options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < fx:
                    x, fx = res.x, res.fun
            except Exception:
                pass # Keep previous best if polish fails

        return x, fx
# EVOLVE-BLOCK-END
