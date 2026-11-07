"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-Start Racing Optimizer for Weierstrass f16.
    
    Strategy:
    1. Split the total budget (500) into K independent "races" (e.g., K=10, 50 calls each).
    2. Each race starts from a unique random point in [-4, 4]^5.
    3. Each race uses a fast, adaptive local search (1+1-ES with 1/5 rule) to rapidly 
       descend into the nearest basin.
    4. The global best is tracked across all races.
    5. This ensures early progress (high AOCC) because multiple independent local searches 
       are running in parallel. If any one finds a good basin, the score improves quickly.
       It also avoids the risk of a single global optimizer wasting all 500 calls in a 
       poor region.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Safety margin for time (evaluator kills at 330s, we stop at 295s)
        start_time = time.time()
        deadline = start_time + 295.0
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Wrapper to count calls, enforce time/budget, and clip
        def wrapped_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                return best_f
            if calls_made >= budget:
                return best_f
            
            # Ensure input is a valid numpy array
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            
            try:
                f_val = func(x)
            except Exception:
                f_val = np.inf
            
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # --- Multi-Start Configuration ---
        # We use 10 independent starts.
        # Budget per start: 50 calls.
        # This allows 10 * 50 = 500 calls total.
        
        num_starts = 10
        calls_per_start = budget // num_starts
        
        # Set a fixed seed for reproducibility within the run if needed, 
        # but random is fine as the evaluator handles instances.
        # Using a local RNG to ensure independence of starts if possible, 
        # though global np.random is fine.
        
        for start_idx in range(num_starts):
            # Check time and budget before starting a new race
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Initialize this race's local search
            # Start point: Uniform in [-4, 4] to cover the likely minimizer range
            x = np.random.uniform(-4, 4, dim)
            
            # Initial step size: 
            # Weierstrass has features at multiple scales. 
            # A moderate step size allows quick local convergence.
            # 0.5 is a reasonable initial scale for dim=5 in [-5,5].
            sigma = 0.5
            
            # 1/5 Rule parameters
            # We maintain a history of successful steps to adapt sigma.
            # Simple implementation: track success/failure of last 5 steps?
            # For speed and simplicity in a short budget (50 calls), 
            # we can use a simpler adaptive rule or fixed sigma with restarts.
            # Let's use a simple adaptive 1+1-ES.
            
            current_x = x
            current_f = wrapped_func(current_x)
            
            # Local search loop for this start
            local_calls = 0
            max_local_calls = calls_per_start - 1 # 1 already used for init
            
            while local_calls < max_local_calls:
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Generate a candidate step
                # Gaussian perturbation
                step = np.random.normal(0, sigma, dim)
                candidate_x = np.clip(current_x + step, -5, 5)
                
                # Evaluate candidate
                candidate_f = wrapped_func(candidate_x)
                local_calls += 1
                
                # Update best for this local search
                if candidate_f < current_f:
                    # Success
                    current_x = candidate_x
                    current_f = candidate_f
                    # Increase sigma slightly (1/5 rule: increase by 1.1 on success)
                    sigma *= 1.1
                else:
                    # Failure
                    # Decrease sigma (1/5 rule: decrease by 1/1.1 on failure)
                    sigma /= 1.1
                    
                # Early termination for this start if step size becomes too small
                # or if we haven't improved for a while?
                # With only 50 calls, we just run them all to be safe.
                
                # Optional: If sigma gets very small, we might be stuck.
                # But with only 50 calls, we don't have much room to restart locally.
                # The multi-start nature handles this by moving to the next start.

        return best_x, best_f
# EVOLVE-BLOCK-END
