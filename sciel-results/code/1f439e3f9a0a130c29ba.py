"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale coordinate-wise search with strategic restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(123)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Strategy: 
        # 1. Use a deterministic coordinate-wise hill climber with geometric step decay.
        # 2. Perform this from multiple strategic starting points to handle the rotation/shift.
        # 3. The "Lunacek" function has two basins. One is global, one is local. 
        #    They are separated by a significant distance. 
        #    A multi-scale search allows "jumping" over the barrier if the step size is large enough,
        #    or crawling along the valley.
        
        # Define the step schedule. 
        # We want to cover the box [-5, 5]. 
        # Start step = 5.0. 
        # Decay factor = 0.8. 
        # Number of steps such that we use ~80% of budget per restart if needed, 
        # but we will split budget among restarts.
        
        # Let's define a function that runs the coordinate search given a starting point and remaining budget.
        def coordinate_search(x_start, max_calls):
            nonlocal calls_made
            x = x_start.copy()
            fx = evaluate(x)
            if calls_made >= budget or max_calls <= 0:
                return x, fx
            
            # Step sizes: 5.0, 4.0, 3.2, ... down to very small
            # 5.0 * 0.8^k
            # We need to ensure we don't exceed max_calls.
            # Each "level" of step size costs 2 * dim calls (try + and - for each dim).
            # So max_levels = max_calls // (2 * dim)
            
            # But we can be smarter: if we improve, we might want to stay at this step size longer?
            # No, for determinism and simplicity, we fix the sequence.
            
            step = 5.0
            decay = 0.8
            min_step = 1e-4
            dim = self.dim
            
            # Pre-calculate how many levels we can afford
            # Actually, let's just loop and check budget.
            
            while step > min_step:
                improved = False
                for i in range(dim):
                    if calls_made >= budget or (max_calls - (calls_made - (calls_made - max_calls))) >= max_calls: # This logic is messy
                        break
                    
                    # Try + step
                    x_trial = x.copy()
                    x_trial[i] += step
                    if np.any(x_trial < -5) or np.any(x_trial > 5):
                        # Clip or skip? Clipping is safer to stay in bounds, but might be less effective.
                        # The func clips anyway.
                        x_trial = np.clip(x_trial, -5, 5)
                    
                    fx_trial = evaluate(x_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                        # Note: we do not repeat this direction immediately in this simple version.
                        # We just accept and move to next dim.
                    
                    # Try - step
                    if calls_made >= budget:
                        break
                        
                    x_trial = x.copy()
                    x_trial[i] -= step
                    x_trial = np.clip(x_trial, -5, 5)
                    fx_trial = evaluate(x_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                
                if not improved:
                    # If no improvement at this step size, we can shrink step size.
                    # In some implementations, you might repeat the step size if improvement occurred.
                    # Here we always shrink to ensure progress in "resolution".
                    pass
                
                step *= decay
                
                if time.time() > deadline:
                    break
            
            return x, fx

        # Strategic Starting Points:
        # 1. Center (0,0,0,0,0)
        # 2. A point biased to one side (e.g., -3, 0, 0, 0, 0) to catch the "other" basin
        # 3. A point biased to the other side (e.g., 3, 0, 0, 0, 0)
        # 4. A random point
        
        # We split the budget. 
        # Let's give 30% to each of the first 3, and 10% to the last.
        # Or simply run them sequentially, stopping if budget is low.
        
        starts = [
            np.zeros(dim),
            np.array([-3.0, 0, 0, 0, 0]) if dim >= 1 else np.zeros(dim),
            np.array([3.0, 0, 0, 0, 0]) if dim >= 1 else np.zeros(dim),
            rng.uniform(-5, 5, dim)
        ]
        
        # Ensure starts are valid
        for i in range(len(starts)):
            starts[i] = np.clip(starts[i], -5, 5)
        
        # Budget allocation
        # Total 500.
        # 150, 150, 100, 100
        allocs = [150, 150, 100, 100]
        
        for i, start in enumerate(starts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # If this is the last restart, use all remaining budget
            if i == len(starts) - 1:
                alloc = budget - calls_made
            else:
                alloc = min(allocs[i], budget - calls_made)
            
            if alloc <= 0:
                continue
                
            # Run the coordinate search
            # We pass 'alloc' as a soft limit, but the function checks 'budget' (global) 
            # and we can check 'calls_made' against a local limit.
            # Let's modify coordinate_search to take a hard limit on calls *for this restart*.
            
            def run_limited_search(x_start, limit):
                nonlocal calls_made
                local_start_calls = calls_made
                x, fx = coordinate_search(x_start, limit)
                return x, fx
            
            # The coordinate_search above doesn't strictly enforce 'max_calls' in the loop condition correctly 
            # because it checks 'calls_made >= budget'. 
            # Let's refactor to ensure we don't bleed into the next restart's budget too much.
            
            # Actually, the previous `coordinate_search` has a flaw: it doesn't stop based on `max_calls` passed to it, 
            # only on global `budget`. 
            # Let's rewrite the inner loop to be cleaner.
            
            x = start.copy()
            fx = evaluate(x)
            
            step = 5.0
            decay = 0.8
            min_step = 1e-4
            d = dim
            
            # Calculate max steps we can afford: limit // (2 * d)
            # But we might break early if no improvement.
            
            while step > min_step:
                if calls_made >= budget or (calls_made - local_start_calls) >= alloc:
                    break
                if time.time() > deadline:
                    break
                
                improved = False
                for i in range(d):
                    if calls_made >= budget or (calls_made - local_start_calls) >= alloc:
                        break
                    
                    # + step
                    x_trial = x.copy()
                    x_trial[i] += step
                    x_trial = np.clip(x_trial, -5, 5)
                    fx_trial = evaluate(x_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                    
                    # - step
                    if calls_made >= budget or (calls_made - local_start_calls) >= alloc:
                        break
                    x_trial = x.copy()
                    x_trial[i] -= step
                    x_trial = np.clip(x_trial, -5, 5)
                    fx_trial = evaluate(x_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                
                # Shrink step regardless of improvement? 
                # If improved, maybe keep step size? 
                # Standard coordinate descent often keeps step size if improvement to "walk" the valley.
                # Let's try: if improved, keep step. Else, shrink.
                if not improved:
                    step *= decay
                else:
                    # Optional: slight shrink to ensure progress? 
                    # No, keep it to allow large moves in the right direction.
                    pass
                
                # Safety: if step becomes very small and no improvement for a while, break?
                # The min_step check handles this.
                
            # After search, update best (already done in evaluate)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
