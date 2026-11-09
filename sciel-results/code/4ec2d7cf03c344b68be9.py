import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin of 10s
        
        # 1. Initialization with Latin Hypercube Sampling (LHS)
        # This ensures good initial spread and early progress for AOCC
        n_init = 10
        x_init = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            # LHS: divide [0,1] into n_init intervals, pick one random point in each
            perm = np.random.permutation(n_init)
            x_init[:, i] = (perm + np.random.rand(n_init)) / n_init
        
        # Map to [-5, 5]
        x_init = x_init * 10.0 - 5.0
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Evaluate initial points
        for x in x_init:
            if time.time() > deadline:
                break
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no calls made
            x = np.zeros(self.dim)
            best_f = func(x)
            best_x = x.copy()
            calls_made = 1

        # 2. Main Optimization Loop: (1+1)-ES with Restart Mechanism
        # Start from the best initial point
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size
        
        stagnation_counter = 0
        max_stagnation = 20  # Restart if no improvement for 20 steps
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            # Add some noise
            y = x + sigma * np.random.randn(self.dim)
            # Clip to bounds
            y = np.clip(y, -5.0, 5.0)
            
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                # Improvement
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                stagnation_counter = 0
                # Increase step size slightly on success (adaptive)
                sigma *= 1.2
                # Cap sigma to avoid jumping out of basin
                if sigma > 5.0:
                    sigma = 5.0
            else:
                # No improvement
                stagnation_counter += 1
                # Decrease step size
                sigma *= 0.8
                # Floor for sigma
                if sigma < 1e-4:
                    sigma = 1e-4
                
                # Restart Logic
                if stagnation_counter >= max_stagnation:
                    # Restart from global best with a larger sigma to escape local minima
                    x = best_x.copy()
                    fx = best_f
                    sigma = 2.0 + np.random.rand() * 2.0 # Randomized restart size
                    stagnation_counter = 0
                    
                    # If we have been stuck for a long time, maybe try a new random point?
                    # For now, restarting from global best is safer for Katsuura
                    # If we want more exploration, we could pick a new LHS point here,
                    # but restarting from best with larger sigma is the proven strategy.
                    
                    # Optional: If fx is still very high, consider a new random start
                    # But let's stick to the proven "restart from global best"
        
        return best_x, best_f
