"""Combined Optimizer: Fixed-Phase Multi-Restart (1+1)-ES with Adaptive Sigma."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB-style Rastrigin (f15) using a fixed-phase strategy.
    
    Strategy:
    1. Phase 1: Explore from the origin (high prior for center).
    2. Phase 2: Explore from a random central point (diversity).
    3. Phase 3: Refine the global best found so far (precision).
    
    Uses (1+1)-ES with window-based sigma adaptation for each phase.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        n = self.dim
        total_budget = self.budget
        
        best_x_global = None
        best_f_global = float('inf')
        
        calls_made = 0
        
        # Fixed seed for reproducibility within the instance, 
        # but different from previous attempts to ensure diversity
        rng = np.random.default_rng(123)
        
        # Budget Allocation:
        # Phase 1: 30% (Origin)
        # Phase 2: 30% (Random Central)
        # Phase 3: 40% (Polish Global Best)
        phase1_budget = int(total_budget * 0.30)
        phase2_budget = int(total_budget * 0.30)
        phase3_budget = total_budget - phase1_budget - phase2_budget
        
        # Parameters for (1+1)-ES
        initial_sigma = 2.0
        target_success_rate = 0.2
        window_size = 10
        stagnation_limit = 30  # Slightly lower to allow phase transitions
        
        def run_es_phase(start_x, max_calls):
            """
            Runs (1+1)-ES from start_x for up to max_calls.
            Returns the best x and f found in this phase.
            """
            nonlocal best_x_global, best_f_global, calls_made
            
            if calls_made >= total_budget:
                return start_x, float('inf')
                
            x = start_x.copy()
            # Ensure start_x is evaluated
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
            
            # Run (1+1)-ES for this phase
            while calls_made < total_budget and calls_made < (start_time + 0): # Placeholder for logic
                pass # Loop logic below
            
            # Corrected loop structure
            local_calls_used = 1 # Initial eval
            while local_calls_used < max_calls and calls_made < total_budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                z = np.random.standard_normal(n)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = func(x_trial)
                calls_made += 1
                local_calls_used += 1
                
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
                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x.copy()
                else:
                    # Failure
                    stagnation_counter += 1
                    
                # Update success rate window
                window.append(1 if f_trial < fx else 0)
                if len(window) > window_size:
                    window.pop(0)
                
                # Adapt sigma based on 1/5 rule approximation
                if len(window) >= window_size:
                    current_rate = np.mean(window)
                    if current_rate > target_success_rate:
                        sigma *= 1.15
                    else:
                        sigma /= 1.15
                    # Reset window to avoid bias from past history
                    window = []
                
                # Clamp sigma to reasonable bounds
                if sigma < 1e-5:
                    sigma = 1e-5
                if sigma > 15.0:
                    sigma = 15.0
                    
                # Check for stagnation
                if stagnation_counter >= stagnation_limit:
                    break
            
            return local_best_x, local_best_f

        # Phase 1: Start from origin
        if calls_made < total_budget:
            start1 = np.zeros(n)
            run_es_phase(start1, phase1_budget)
            
        # Phase 2: Start from a random point in the central region
        if calls_made < total_budget:
            start2 = rng.uniform(-3, 3, n)
            run_es_phase(start2, phase2_budget)
            
        # Phase 3: Polish the best solution found so far
        if calls_made < total_budget:
            if best_x_global is not None:
                start3 = best_x_global.copy()
            else:
                start3 = rng.uniform(-5, 5, n)
            run_es_phase(start3, phase3_budget)
            
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
