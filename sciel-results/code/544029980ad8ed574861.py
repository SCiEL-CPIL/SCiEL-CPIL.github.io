"""Rewritten optimizer using Multi-Restart (1+1)-ES for Rastrigin rotated f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES for global optimization on rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        n = self.dim
        total_budget = self.budget
        
        best_x_global = None
        best_f_global = float('inf')
        
        calls_made = 0
        
        # Parameters for (1+1)-ES
        # Initial step size
        initial_sigma = 2.0 
        # Success rate target for 1/5 rule
        target_success_rate = 0.2
        # Window for success rate calculation
        window_size = 10
        # Stagnation limit: how many evaluations without improvement to restart
        stagnation_limit = 40
        
        # We will perform multiple restarts. 
        # The number of restarts is dynamic based on budget and stagnation.
        
        while calls_made < total_budget:
            if time.time() > deadline:
                break
                
            # Start a new restart
            # Initialize x randomly in the box
            x = np.random.uniform(-5, 5, n)
            fx = func(x)
            calls_made += 1
            
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()
                
            sigma = initial_sigma
            success_count = 0
            window = []
            stagnation_counter = 0
            local_best_x = x.copy()
            local_best_f = fx
            
            # Run (1+1)-ES for this restart
            while calls_made < total_budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                # Sample direction from standard normal
                z = np.random.standard_normal(n)
                x_trial = x + sigma * z
                # Clip to box
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = func(x_trial)
                calls_made += 1
                
                # Update global best
                if f_trial < best_f_global:
                    best_f_global = f_trial
                    best_x_global = x_trial.copy()
                    
                if f_trial < fx:
                    # Success
                    x = x_trial
                    fx = f_trial
                    success_count += 1
                    stagnation_counter = 0
                else:
                    # Failure
                    stagnation_counter += 1
                    
                # Update success rate window
                window.append(1 if f_trial < fx else 0)
                if len(window) > window_size:
                    window.pop(0)
                
                # Adapt sigma based on 1/5 rule approximation
                # If success rate > target, increase sigma; else decrease
                if len(window) >= window_size // 2:
                    current_rate = np.mean(window)
                    if current_rate > target_success_rate:
                        sigma *= 1.25
                    else:
                        sigma /= 1.25
                    # Reset window to avoid bias from past history
                    window = []
                
                # Clamp sigma to reasonable bounds
                if sigma < 1e-4:
                    sigma = 1e-4
                if sigma > 10.0:
                    sigma = 10.0
                    
                # Check for stagnation
                if stagnation_counter >= stagnation_limit:
                    break
            
            # If we exited due to stagnation or time, loop back to start a new restart
            # If we exited due to budget, the outer loop will terminate
            
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
