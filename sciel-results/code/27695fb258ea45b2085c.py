"""Rewritten program with Adaptive Coordinate Descent for anisotropic functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using Adaptive Coordinate Descent.
    
    This method is specifically designed for anisotropic functions like the Bent Cigar,
    where step sizes need to vary significantly between dimensions. It maintains
    independent step sizes for each dimension and adapts them based on success/failure
    of coordinate moves.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit, stop at 290s
        
        rng = np.random.default_rng(42)
        
        # Initial point: Random uniform in the box
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Initial step sizes: Start moderately large
        sigmas = np.full(self.dim, 1.0)
        
        calls_made = 1
        stagnation_counter = 0
        improvement_threshold = 1e-10
        
        # Main loop
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # If stagnating, try a random restart or large jump
            # For Bent Cigar, coordinate descent is robust, but random restarts help if trapped
            # We use a simple heuristic: if no improvement for many iterations, jump
            if stagnation_counter > 50:
                # Random restart
                x_new = rng.uniform(-5, 5, self.dim)
                fx_new = func(x_new)
                calls_made += 1
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new
                    x = x_new
                    fx = fx_new
                    sigmas = np.full(self.dim, 1.0)
                    stagnation_counter = 0
                else:
                    # Keep current x, but maybe increase sigmas to explore more?
                    # Or just continue.
                    pass
                if calls_made >= self.budget:
                    break

            # Coordinate Descent Step
            improved_this_cycle = False
            
            for i in range(self.dim):
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break
                
                # Try moving in positive direction
                x_trial = x.copy()
                x_trial[i] += sigmas[i]
                x_trial = np.clip(x_trial, -5, 5)
                fx_trial = func(x_trial)
                calls_made += 1
                
                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    sigmas[i] *= 1.5  # Increase step size if successful
                    improved_this_cycle = True
                else:
                    sigmas[i] *= 0.7  # Decrease step size if failed
                    
                    # Try moving in negative direction
                    x_trial_neg = x.copy()
                    x_trial_neg[i] -= sigmas[i]
                    x_trial_neg = np.clip(x_trial_neg, -5, 5)
                    fx_trial_neg = func(x_trial_neg)
                    calls_made += 1
                    
                    if fx_trial_neg < fx:
                        x = x_trial_neg
                        fx = fx_trial_neg
                        sigmas[i] *= 1.5  # Increase step size if successful
                        improved_this_cycle = True
                    else:
                        sigmas[i] *= 0.7  # Decrease step size if failed
                
                # Update global best
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

            if not improved_this_cycle:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
