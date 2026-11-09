"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Restart (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        n = self.dim
        total_budget = self.budget
        
        # Use 4 restarts for a more balanced approach
        n_restarts = 4
        budget_per_restart = total_budget // n_restarts
        
        best_x = None
        best_fx = np.inf
        rng = np.random.default_rng(0)
        
        for r in range(n_restarts):
            # Check time
            if time.time() > deadline:
                break
                
            # Check if we have enough budget left for a meaningful run
            remaining_budget = total_budget - (r * budget_per_restart)
            if remaining_budget <= 0:
                break
            
            # Use slightly less than the full allocated budget to account for overhead
            current_budget = min(budget_per_restart, remaining_budget)
            
            # Initialize
            if r == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-4, 4, n)
            fx = func(x)
            
            # If this start is better than the global best, update
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            # (1+1)-ES loop
            sigma = 1.5
            window_size = 12
            window = []
            
            for i in range(current_budget - 1):
                if time.time() > deadline:
                    break
                    
                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                
                # Update
                if fy < fx:
                    x = y
                    fx = fy
                    # Update global best
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    
                    # On success, slightly increase sigma
                    window.append(1)
                else:
                    window.append(0)
                    
                # Maintain sliding window
                if len(window) > window_size:
                    window.pop(0)
                
                # Adjust sigma based on success rate in window
                if len(window) == window_size:
                    success_rate = sum(window) / window_size
                    if success_rate > 0.3:
                        sigma *= 1.15
                    elif success_rate < 0.15:
                        sigma *= 0.85
                    
                # Prevent sigma from exploding or vanishing
                if sigma > 8.0: 
                    sigma = 8.0
                if sigma < 1e-10: 
                    sigma = 1e-10
                    
        return best_x, best_fx
# EVOLVE-BLOCK-END
