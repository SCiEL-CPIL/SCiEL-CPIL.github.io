"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Adaptive (1+1)-ES with Best-First Selection and Global Restarting."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We use 320s to be safe.
        deadline = start_time + 320.0
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def check_time():
            return time.time() > deadline

        # Parameters
        # Initial step size: cover the domain [-5, 5] reasonably well
        # With dim=5, a step of 2.0 gives a good spread
        init_step = 2.0
        
        # 1/5 Rule parameters
        window_size = 10
        success_count = 0
        successes_in_window = 0
        
        # Restart logic
        stagnation_limit = 50 # If no improvement for 50 steps, restart from global best
        steps_since_improvement = 0
        
        # Main Loop
        x = rng.uniform(-4, 4, dim)
        f = func(x)
        calls_made += 1
        update_best(x, f)
        
        step_size = init_step
        
        while calls_made < budget and not check_time():
            # Generate candidate
            z = rng.standard_normal(dim)
            x_trial = x + step_size * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls_made += 1
            
            if f_trial < f:
                # Success
                x = x_trial
                f = f_trial
                success_count += 1
                successes_in_window += 1
                steps_since_improvement = 0
                update_best(x, f)
            else:
                # Failure
                # We keep x, but we track stagnation
                steps_since_improvement += 1
            
            # Adapt step size every 'window_size' steps
            if success_count % window_size == 0:
                # 1/5 Rule: target success rate is 0.2
                # If success rate > 0.2, increase step size
                # If success rate < 0.2, decrease step size
                rate = successes_in_window / window_size
                
                if rate > 0.2:
                    step_size *= 1.5
                elif rate < 0.2:
                    step_size /= 1.5
                # else keep same
                
                # Clamp step size
                step_size = np.clip(step_size, 0.01, 10.0)
                
                successes_in_window = 0
            
            # Restart logic: If we haven't improved in 'stagnation_limit' steps,
            # jump back to the global best and reset local state
            if steps_since_improvement > stagnation_limit:
                x = best_x.copy()
                f = best_f
                steps_since_improvement = 0
                # Reset step size to a moderate value to explore around best
                step_size = init_step * 0.5 
                # Optional: Add a small random perturbation to avoid exact same trajectory
                x = x + rng.uniform(-0.1, 0.1, dim)
                x = np.clip(x, -5, 5)
                # Note: We do NOT evaluate x here to save budget, we just set state.
                # The next iteration will generate a trial from this new x.

        return best_x, best_f

# EVOLVE-BLOCK-END
