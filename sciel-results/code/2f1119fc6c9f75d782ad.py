import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def safe_func(x):
            nonlocal calls
            if calls >= budget:
                return np.inf # Should not be called
            calls += 1
            return func(x)
            
        # (1+4)-ES with restarts
        # We will run several restarts.
        # Each restart runs until a stagnation limit or budget is low.
        
        # Initial parameters
        lambda_ = 4
        sigma = 1.0
        min_sigma = 1e-4
        
        for restart in range(10):
            if calls >= budget or time.time() > deadline:
                break
            
            # Initialize new point
            x = rng.uniform(-5.0, 5.0, dim)
            f_curr = safe_func(x)
            calls += 1 # Wait, safe_func increments. So calls is 1 after this.
            
            if f_curr < best_f:
                best_f = f_curr
                best_x = x.copy()
                
            if calls >= budget:
                break
                
            # Local search phase
            # Run (1+4)-ES from this point
            # We will run for a fixed number of steps or until stagnation
            # Let's run for 50 steps (200 calls) per restart.
            # 10 restarts * 200 = 2000 calls? No, budget is 500.
            # So we have 500 calls total.
            # Let's allocate 50 calls per restart. 10 restarts.
            
            steps_in_restart = 0
            max_steps_in_restart = 50
            stagnation_counter = 0
            local_best_f = f_curr
            
            while steps_in_restart < max_steps_in_restart:
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate 4 candidates
                best_candidate_x = None
                best_candidate_f = np.inf
                
                for k in range(lambda_):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Gaussian step
                    direction = rng.normal(0, 1, dim)
                    x_new = x + sigma * direction
                    x_new = np.clip(x_new, -5.0, 5.0)
                    
                    f_new = safe_func(x_new)
                    
                    if f_new < best_candidate_f:
                        best_candidate_f = f_new
                        best_candidate_x = x_new
                        
                if best_candidate_x is None:
                    break
                
                # Update
                if best_candidate_f < local_best_f - 1e-12:
                    x = best_candidate_x
                    f_curr = best_candidate_f
                    local_best_f = best_candidate_f
                    # Increase sigma slightly on success?
                    sigma *= 1.05
                    stagnation_counter = 0
                else:
                    # No improvement
                    sigma *= 0.9
                    stagnation_counter += 1
                    
                    # If sigma is too small, restart this local search?
                    if sigma < min_sigma:
                        break
                        
                    # If stagnated too long, break to restart
                    if stagnation_counter > 10:
                        break
                        
                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x.copy()
                    
                steps_in_restart += 1
                
        return best_x, best_f
