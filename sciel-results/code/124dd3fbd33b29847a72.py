import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Keep a safety margin. 300s limit, 330s kill. 
        # We will use 290s as our internal deadline to be safe.
        self.deadline = time.time() + 290

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        # Start at origin, which is usually close enough or in the valley
        # The problem states minimizer is in [-4, 4]^5
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Initialize separate step sizes
        # Start with a moderate step size. 
        # Bent cigar has one stiff dim and others soft.
        # A uniform start is fine, adaptation will fix it.
        sigma = np.full(dim, 0.5)
        
        # Adaptation parameters
        # For (1+1)-ES, typical update is:
        # if improved: sigma *= c_up
        # else: sigma *= c_down
        # c_up ~ 1.2, c_down ~ 1/1.2
        c_up = 1.1
        c_down = 1.0 / 1.1
        
        # Restart logic
        stagnation_counter = 0
        max_stagnation = 30 # If no improvement in 30 steps, restart
        restart_count = 0
        max_restarts = 5
        
        # Main Evolutionary Strategy Loop
        while calls < budget and time.time() < self.deadline:
            # Generate candidate
            # Sample from N(0, sigma_i) for each dim
            # We use rng.normal with size=dim and scale=sigma
            # Note: rng.normal(0, sigma) where sigma is an array works in NumPy
            mutation = rng.normal(0, 1, dim) * sigma
            x_new = x + mutation
            x_new = np.clip(x_new, -5, 5)
            
            f_new = func(x_new)
            calls += 1
            
            if f_new < fx:
                # Improvement
                x = x_new
                fx = f_new
                best_x = x.copy()
                best_f = fx
                stagnation_counter = 0
                
                # Increase step sizes
                sigma *= c_up
                # Prevent sigma from growing too large
                sigma = np.clip(sigma, 1e-8, 10.0)
            else:
                # No improvement
                stagnation_counter += 1
                # Decrease step sizes
                sigma *= c_down
                # Prevent sigma from becoming too small
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Check for stagnation
                if stagnation_counter >= max_stagnation:
                    restart_count += 1
                    if restart_count > max_restarts:
                        break
                    
                    # Restart: Keep best_x but increase step size to escape
                    # We might be in a very flat area or stuck due to tiny steps
                    # Reset sigma to a larger value to explore again
                    sigma = np.full(dim, 0.5)
                    # Optionally, jump slightly from best_x?
                    # No, keep x at best_x, just increase exploration power
                    # Actually, if we are stuck, it means gradient is ~0 or we are in a flat valley.
                    # Increasing sigma allows us to probe further.
                    stagnation_counter = 0
                    max_stagnation = int(max_stagnation * 1.5) # Increase patience after restart
        
        # Phase 2: Coordinate Descent Polish
        # The ES might stop with steps still slightly too large for final precision.
        # CD is very efficient for separable or nearly separable functions like Bent Cigar.
        if calls < budget and time.time() < self.deadline:
            # Initialize CD step sizes based on current sigma, but smaller
            cd_sigma = sigma.copy() * 0.1
            # Ensure minimum step
            cd_sigma = np.maximum(cd_sigma, 1e-5)
            
            x_cd = best_x.copy()
            fx_cd = best_f
            
            no_progress = 0
            max_cd_iterations = 1000 # Cap iterations to avoid infinite loop if stuck
            
            while calls < budget and time.time() < self.deadline and no_progress < 20:
                progress_made = False
                for i in range(dim):
                    if calls >= budget or time.time() > self.deadline:
                        break
                    
                    # Try positive step
                    x_try = x_cd.copy()
                    x_try[i] += cd_sigma[i]
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls += 1
                    
                    if f_try < fx_cd:
                        x_cd = x_try
                        fx_cd = f_try
                        progress_made = True
                        # Keep step or increase slightly if we are making good progress?
                        # Standard CD usually keeps step if successful, halves if not.
                        # But for anisotropic, we want to shrink stiff dim fast.
                        # Let's stick to: if success, keep step. If fail, shrink.
                    else:
                        # Try negative step
                        x_try = x_cd.copy()
                        x_try[i] -= cd_sigma[i]
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        calls += 1
                        
                        if f_try < fx_cd:
                            x_cd = x_try
                            fx_cd = f_try
                            progress_made = True
                        else:
                            # No progress in this dimension, shrink step
                            cd_sigma[i] *= 0.5
                            if cd_sigma[i] < 1e-8:
                                cd_sigma[i] = 1e-8

                if progress_made:
                    no_progress = 0
                    # Update global best
                    if fx_cd < best_f:
                        best_f = fx_cd
                        best_x = x_cd.copy()
                else:
                    no_progress += 1
                    # If no progress in a full cycle, shrink all steps further
                    cd_sigma *= 0.5
                    if np.all(cd_sigma < 1e-8):
                        break

        return best_x, best_f
