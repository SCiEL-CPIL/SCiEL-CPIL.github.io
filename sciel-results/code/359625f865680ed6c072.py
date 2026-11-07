"""Multi-start (1+1)-ES optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using Multi-start (1+1)-ES with 1/5 success rule.
    This approach is lightweight and effective for small budgets like 500 calls,
    leveraging restarts to escape local minima and explore the search space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We aim to finish well before.
        # 280s is a safe margin.
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0

        # Strategy: Multi-start (1+1)-ES
        # We divide the budget into several segments.
        # If a segment doesn't find the global minimum (or makes very little progress), we restart.
        # For 500 calls, let's try segments of ~50 calls.
        # This allows ~10 restarts.
        
        segment_size = 50
        max_segments = budget // segment_size + 1
        
        # Initial sigma for (1+1)-ES
        # A reasonable initial step size for [-5, 5]^dim is 1.0 or 2.0
        # We will adapt it using the 1/5 success rule.
        
        for seg in range(max_segments):
            # Check time
            if time.time() > deadline - 10.0:
                break
            if calls >= budget:
                break

            # Initialize new start point
            # Random uniform in [-5, 5]
            x = rng.uniform(-5, 5, dim)
            sigma = 1.0
            
            # Evaluate initial point
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Track progress in this segment
            seg_best_f = fx
            seg_best_x = x.copy()
            seg_calls = 1
            success_count = 0
            total_attempts = 0
            
            # Run (1+1)-ES for this segment
            while seg_calls < segment_size:
                # Check global time
                if time.time() > deadline - 5.0:
                    break
                if calls >= budget:
                    break
                
                # Generate candidate
                direction = rng.standard_normal(dim)
                y = x + sigma * direction
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                seg_calls += 1
                total_attempts += 1
                
                # Update global best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                # Update segment best and check for success
                if fy < seg_best_f:
                    success_count += 1
                    seg_best_f = fy
                    seg_best_x = y.copy()
                    # Accept step
                    x = y
                else:
                    # Reject step
                    pass
                
                # Adapt sigma using 1/5 success rule
                # We update sigma periodically or every step? 
                # Standard 1/5 rule: if success rate > 1/5, increase sigma, else decrease.
                # To be stable, we can update every few steps or use a running average.
                # For simplicity and speed, we update every step but with a small learning rate.
                # Or, we can batch updates. Let's do a simple update.
                # A common variant:
                # if success_count / total_attempts > 0.2: sigma *= 1.05
                # else: sigma /= 1.05
                
                # More robust: Update every 10 attempts
                if total_attempts % 10 == 0:
                    success_rate = success_count / total_attempts
                    if success_rate > 0.2:
                        sigma *= 1.05
                    else:
                        sigma /= 1.05
                    
                    # Reset counters for the next batch? 
                    # Usually, the 1/5 rule uses a longer window or resets.
                    # Let's reset to keep it responsive.
                    success_count = 0
                    total_attempts = 0
                    
                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 10.0)

            # End of segment. 
            # Optional: If we found a new global best in this segment, we might continue 
            # from the best point found in this segment for the next segment, 
            # or we could restart randomly. 
            # Multi-start usually implies random restarts to escape local minima.
            # However, if we found a very good point, it's wasteful to discard it.
            # Let's do a hybrid: 
            # If the segment best is significantly better than the previous global best, 
            # start the next segment from this new best. 
            # Otherwise, start randomly.
            
            # For simplicity and to ensure exploration (as per inspiration), 
            # we will always start the next segment randomly, 
            # BUT we keep the global best. 
            # Actually, if we find a local minimum, random restart is key.
            # If we find the global minimum, we will keep improving it in subsequent segments 
            # if we start near it? No, random start is far.
            # Let's try: If progress was made, continue from best_x. If not, random.
            
            # Let's stick to the inspiration: "Multi-start restarts escape local minima".
            # This suggests random restarts are the key mechanism.
            # I will use random restarts for every segment.

        return best_x, best_f
# EVOLVE-BLOCK-END
