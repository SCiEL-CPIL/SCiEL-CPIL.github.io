import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        # Multi-restart (1+1)-ES strategy
        # This is better for rugged landscapes like Schwefel.
        # We divide the budget into several restarts.
        # Each restart uses a portion of the budget to explore from a new starting point.
        
        num_restarts = 10
        budget_per_restart = budget // num_restarts
        # Ensure we don't exceed budget
        # If budget_per_restart is too small, increase num_restarts logic or just use remaining budget in last restart.
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Determine budget for this restart
            if r < num_restarts - 1:
                current_budget = budget_per_restart
            else:
                current_budget = budget - calls_made
            
            if current_budget <= 0:
                break
            
            # Initialize for this restart
            # Start from a random point in [-4, 4]^d to avoid boundaries
            mean = rng.uniform(-4, 4, dim)
            sigma = 2.0  # Initial step size
            
            # (1+1)-ES loop
            local_calls = 0
            local_best_f = np.inf
            local_best_x = mean.copy()
            
            # First evaluation
            f_curr = func(mean)
            local_calls += 1
            calls_made += 1
            local_best_f = f_curr
            local_best_x = mean.copy()
            
            while local_calls < current_budget:
                if time.time() > deadline:
                    break
                    
                # Generate mutant
                z = rng.standard_normal(dim)
                mutant = mean + sigma * z
                mutant = np.clip(mutant, -5, 5)
                
                # Evaluate mutant
                f_mutant = func(mutant)
                local_calls += 1
                calls_made += 1
                
                # Update best
                if f_mutant < local_best_f:
                    local_best_f = f_mutant
                    local_best_x = mutant.copy()
                    # Success: increase sigma slightly? Or keep?
                    # Standard (1+1)-ES: if success, sigma *= c_up, else sigma *= c_down
                    # c_up = 1 + 1/(dim+1), c_down = 1 - 1/(dim+1)
                    sigma *= 1.0 + 1.0 / (dim + 1.0)
                else:
                    # Failure: decrease sigma
                    sigma *= 1.0 - 1.0 / (dim + 1.0)
                
                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update mean to the better point (accept/reject)
                if f_mutant < f_curr:
                    mean = mutant.copy()
                    f_curr = f_mutant
                # else: mean stays the same, f_curr stays the same (reject mutant)
                
                # Update global best
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()
                    
            # End of restart
            # If we found a very good solution, we could stop early, but let's use full budget.
            
        return best_x, best_f
