"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+λ)-ES with adaptive step size and time guard for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        n = self.dim
        calls = [0]
        best_fx = np.inf
        best_x = np.zeros(n) # Initialize to avoid None

        def wrapper(x):
            if calls[0] >= self.budget:
                return best_fx
            calls[0] += 1
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Multi-start Nelder-Mead
        # 5D, 500 calls.
        # 10 restarts, 40 calls each.
        n_restarts = 10
        max_iter_per_restart = 40
        
        rng = np.random.default_rng(42)
        
        for i in range(n_restarts):
            if time.time() > deadline:
                break
            if calls[0] >= self.budget:
                break
            
            # Initialize center
            x0 = rng.uniform(-5, 5, n)
            
            # Nelder-Mead options
            # maxiter is the number of function evaluations allowed in this run
            # We set it to remaining budget or max_iter_per_restart
            remaining = self.budget - calls[0]
            current_max_iter = min(max_iter_per_restart, remaining)
            
            if current_max_iter <= 0:
                break

            try:
                res = minimize(wrapper, x0, method='Nelder-Mead', 
                               options={'maxiter': current_max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
            except Exception:
                continue

        return best_x, best_fx
# EVOLVE-BLOCK-END
