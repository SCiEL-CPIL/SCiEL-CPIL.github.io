# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive coordinate descent with restarts for sharp ridge functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Initialize at center
        x0 = np.zeros(n)
        f0 = wrapped_func(x0)
        current_x = x0.copy()
        current_f = f0
        
        # Adaptive step sizes for each coordinate
        step_sizes = np.ones(n) * 1.0
        min_step = 1e-6
        max_step = 10.0
        
        # Parameters
        improvement_threshold = 1e-12
        stall_counter = 0
        max_stall = 30
        
        while calls_made[0] < budget and time.time() < deadline:
            improved_this_round = False
            
            # Try each coordinate in random order
            coord_order = rng.permutation(n)
            
            for i in coord_order:
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                
                # Try positive and negative steps
                best_coord_f = current_f
                best_coord_step = 0.0
                
                for direction in [1, -1]:
                    trial_x = current_x.copy()
                    trial_x[i] += direction * step_sizes[i]
                    trial_x = np.clip(trial_x, -5, 5)
                    trial_f = wrapped_func(trial_x)
                    
                    if trial_f < best_coord_f - improvement_threshold:
                        best_coord_f = trial_f
                        best_coord_step = direction * step_sizes[i]
                
                if best_coord_step != 0.0:
                    current_x[i] += best_coord_step
                    current_f = best_coord_f
                    improved_this_round = True
                    stall_counter = 0
                    
                    # Adapt step size: increase if improved, decrease if not
                    if best_coord_f < current_f - improvement_threshold * 10:
                        step_sizes[i] = min(step_sizes[i] * 1.5, max_step)
                    else:
                        step_sizes[i] = max(step_sizes[i] * 0.7, min_step)
                else:
                    # No improvement: shrink step size
                    step_sizes[i] = max(step_sizes[i] * 0.5, min_step)
            
            if improved_this_round:
                # Update global best
                if current_f < best_f:
                    best_f = current_f
                    best_x = current_x.copy()
            else:
                stall_counter += 1
                
                # If stalled, try a random restart
                if stall_counter >= max_stall:
                    new_x = rng.uniform(-4, 4, n)
                    new_f = wrapped_func(new_x)
                    
                    if new_f < current_f:
                        current_x = new_x.copy()
                        current_f = new_f
                        # Reset step sizes
                        step_sizes = np.ones(n) * 1.0
                        stall_counter = 0
                    else:
                        # Keep current but try smaller steps
                        step_sizes = np.ones(n) * 0.5
                        stall_counter = 0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
