"""Simulated Annealing with Adaptive Cooling for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin well under 300s/330s
        
        d = self.dim
        budget = self.budget
        
        if budget <= 0:
            return np.zeros(d), np.inf

        # Initial point: center of the box is a safe, high-probability start
        # for BBOB functions which are often centered or shifted within [-5,5]
        x = np.zeros(d)
        f = func(x)
        calls_used = 1
        
        best_x = x.copy()
        best_f = f
        
        # Simulated Annealing Parameters
        # Initial temperature: High enough to accept large uphill moves initially
        T = 10.0
        # Cooling factor: Exponential decay
        alpha = 0.995
        # Minimum temperature
        T_min = 1e-10
        
        # Step size parameters
        # Initial step size: Broad search
        sigma = 4.0
        sigma_min = 1e-6
        # Step size adaptation: Shrink if stuck, expand slightly if improving?
        # For SA, we usually keep sigma fixed or shrink it slowly.
        # Here we shrink sigma as T decreases to refine the search.
        
        # Stagnation counter for potential restarts or parameter reset
        stagnation_count = 0
        max_stagnation = 50
        
        # We will use the remaining budget for the SA loop
        while calls_used < budget:
            if time.time() > deadline:
                break
                
            # Generate candidate
            # Use Gaussian noise scaled by current sigma
            noise = np.random.normal(0, 1, d)
            x_new = x + sigma * noise
            x_new = np.clip(x_new, -5.0, 5.0)
            
            f_new = func(x_new)
            calls_used += 1
            
            # Acceptance criterion
            delta = f_new - f
            if delta < 0:
                # Improvement: Always accept
                x = x_new
                f = f_new
                stagnation_count = 0
                
                # Update global best
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
                # Adapt step size: If we improved, we might be in a good region.
                # Keep sigma or shrink slightly to refine.
                # Standard SA often keeps sigma constant while T drops.
                # But for multimodal landscapes, shrinking sigma helps convergence.
                # Let's link sigma to T: sigma = initial_sigma * (T / T_init)^0.5
                # Or simply shrink sigma slowly.
                sigma *= 0.999 # Very slow shrinkage
            else:
                # Worse: Accept with probability exp(-delta / T)
                if np.random.rand() < np.exp(-delta / T):
                    x = x_new
                    f = f_new
                    stagnation_count += 1
                else:
                    # Reject, stay at current x
                    stagnation_count += 1
            
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min
            
            # If we have been stuck for a long time, we might be in a local minimum
            # or on a plateau. 
            # Strategy: If stagnation is high, do a "restart" or "jump" to escape.
            # Since we are in a single trajectory SA, a restart is risky.
            # Instead, we can temporarily increase T and sigma to escape.
            if stagnation_count > max_stagnation:
                # Boost temperature and step size to escape local trap
                T = max(T, 5.0)
                sigma = max(sigma, 2.0)
                stagnation_count = 0
                max_stagnation = int(max_stagnation * 1.5) # Increase patience for future escapes
        
        return best_x, best_f
# EVOLVE-BLOCK-END
