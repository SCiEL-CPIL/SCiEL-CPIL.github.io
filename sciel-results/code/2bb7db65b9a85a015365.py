"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-restart (1+1)-ES for BBOB-style Schwefel function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Multi-restart (1+1)-ES strategy
        # Use 3 restarts to escape local minima
        num_restarts = 3
        evals_per_restart = budget // num_restarts
        
        best_x_global = np.zeros(dim)
        best_f_global = np.inf
        evals = 0
        
        for restart in range(num_restarts):
            # Random initial point in the search space
            if restart == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, size=dim)
            
            try:
                f_x = func(x)
            except Exception:
                f_x = np.inf
            evals += 1
            
            if f_x < best_f_global:
                best_f_global = f_x
                best_x_global = x.copy()
            
            # (1+1)-ES with 1/5 success rule
            sigma = 2.0
            successes = 0
            attempts = 0
            window_size = 15  # Window for success rate tracking
            
            while evals < budget and time.time() < deadline:
                # Generate candidate
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                np.clip(x_trial, -5, 5, out=x_trial)
                
                try:
                    f_trial = func(x_trial)
                except Exception:
                    f_trial = np.inf
                evals += 1
                
                # Check if candidate is better
                if f_trial < f_x:
                    x = x_trial
                    f_x = f_trial
                    successes += 1
                    
                    if f_x < best_f_global:
                        best_f_global = f_x
                        best_x_global = x.copy()
                else:
                    attempts += 1
                
                # Adapt step size using 1/5 success rule
                attempts_total = successes + attempts
                if attempts_total >= window_size:
                    success_rate = successes / attempts_total
                    if success_rate > 0.2:
                        sigma *= 1.25
                    elif success_rate < 0.2:
                        sigma /= 1.25
                    # Clip sigma to reasonable bounds
                    sigma = np.clip(sigma, 1e-10, 10.0)
                    successes = 0
                    attempts = 0
        
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
