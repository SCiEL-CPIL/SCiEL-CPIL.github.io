"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Local Search Optimizer for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Strategy: Multi-start Hill Climbing with Adaptive Step Sizes
        # We divide the budget into multiple "restarts".
        # Each restart starts from a random point and performs a local search.
        # This is effective for multimodal functions like Schaffers F7 
        # because it explores many local basins rather than trying to 
        # cross them with a single global trajectory.
        
        # Number of restarts: 
        # We want each restart to have enough budget to converge.
        # Let's say each restart gets at least 10-20 function calls.
        # With budget 500, we can afford ~25-50 restarts.
        # Let's aim for ~30 restarts.
        max_restarts = 30
        calls_per_restart = budget // max_restarts
        
        # If calls_per_restart is too small, increase restarts? 
        # No, if it's too small, local search might not converge.
        # If budget is 500 and restarts 30, that's ~16 calls/restart.
        # That's quite short for local search in 5D.
        
        # Alternative: Fewer restarts, more steps per restart.
        # Let's try 10 restarts, ~50 calls each.
        max_restarts = 10
        calls_per_restart = budget // max_restarts
        
        # Initial exploration: One random point
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        calls += 1
        best_x = x0.copy()
        best_f = f0
        
        for r in range(max_restarts):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
                
            # Start point for this restart
            # Mix of: 
            # 1. Near the current global best (exploitation)
            # 2. Random point in the box (exploration)
            # 3. Point derived from previous best with noise
            
            if r == 0:
                # First restart already done above, skip to next logic or just start fresh
                pass
            
            # Choose start strategy
            if r < max_restarts // 2:
                # Exploration: Random start
                x = rng.uniform(-5, 5, dim)
            else:
                # Exploitation: Perturb current best
                # Step size for perturbation: scale with remaining budget or fixed
                # Fixed step size of 1.0 is reasonable for [-5,5] box
                noise = rng.normal(0, 1.0, dim)
                x = best_x + noise
                x = np.clip(x, -5, 5)
            
            # Local Search Phase
            # Simple Coordinate Descent or Nelder-Mead?
            # Nelder-Mead is robust but might be slow if not careful.
            # Let's use a simple adaptive step size hill climbing.
            
            current_x = x.copy()
            current_f = func(current_x)
            calls += 1
            if current_f < best_f:
                best_f = current_f
                best_x = current_x.copy()
            
            step_size = 1.0
            improvement_count = 0
            max_steps_in_restart = calls_per_restart
            
            for _ in range(max_steps_in_restart):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Try moving in each coordinate direction
                found_improvement = False
                for d in range(dim):
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Positive step
                    x_trial = current_x.copy()
                    x_trial[d] += step_size
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1
                    
                    if f_trial < current_f:
                        current_x = x_trial
                        current_f = f_trial
                        found_improvement = True
                        if current_f < best_f:
                            best_f = current_f
                            best_x = current_x.copy()
                    else:
                        # Try negative step
                        x_trial = current_x.copy()
                        x_trial[d] -= step_size
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls += 1
                        
                        if f_trial < current_f:
                            current_x = x_trial
                            current_f = f_trial
                            found_improvement = True
                            if current_f < best_f:
                                best_f = current_f
                                best_x = current_x.copy()
                
                if found_improvement:
                    improvement_count += 1
                    # Keep step size or slightly increase? 
                    # Standard hill climbing keeps it or decreases if stuck.
                    # Let's keep it constant for simplicity, or adapt.
                    # If we improve, we are in a valley.
                else:
                    # No improvement in any coordinate
                    step_size *= 0.7
                    if step_size < 1e-6:
                        break
            
            # End of restart

        return best_x, best_f
# EVOLVE-BLOCK-END
