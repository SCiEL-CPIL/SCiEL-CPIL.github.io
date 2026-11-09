import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        # 1. Generate a set of diverse starting points using Sobol sequence
        # We want enough points to cover the space well. 
        # For dim=5, 32 or 64 points is a good start for a budget of 300.
        n_starts = 32
        # Use scipy's Sobol if available, otherwise fall back to random or Halton-like
        try:
            from scipy.stats import sobol
            # Generate 2^n points, then take the first n_starts
            # sobol requires power of 2 for dimensionality in some versions, but standard usage is fine
            # Actually, simpler: just use np.random.default_rng with a specific strategy or implement a simple Halton
            # Let's stick to a robust random approach with a twist: 
            # We will use a stratified sampling approach.
            pass
        except ImportError:
            pass
        
        # Robust stratified sampling: divide the box into hypercubes and pick a random point in each
        # For dim=5, we can divide each dimension into k intervals.
        # Total points = k^dim. We want ~32 points. 
        # If dim=5, k=2 gives 32 points. This is perfect.
        k = 2
        starts = []
        rng = np.random.default_rng(42)
        for i in range(k ** self.dim):
            # Convert index i to coordinates in base k
            coords = []
            temp_i = i
            for d in range(self.dim):
                coords.append(temp_i % k)
                temp_i //= k
            # Map [0, k-1] to [-5, 5] with some randomness within the cell
            # Cell size is 10/k = 5
            point = np.array([
                -5 + (c + rng.random()) * (10.0 / k) 
                for c in coords
            ])
            starts.append(point)
        
        # Shuffle starts to avoid bias in order
        rng.shuffle(starts)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # We will iterate through the starts.
        # For each start, run a (1+1)-ES until stagnation or budget/time is up.
        # Stagnation limit: if no improvement for `stagnation_limit` steps, restart.
        # We need to allocate budget carefully.
        # With 32 starts and 300 budget, we can't give 300 to each.
        # We will use a dynamic allocation: 
        # The first few starts get more budget to find a good basin.
        # Or, simpler: run (1+1)-ES from start i until stagnation, then move to start i+1.
        # If we run out of starts, we can restart from the best point found so far with a larger step size (simulated annealing-like).
        
        stagnation_limit = 15
        max_steps_per_restart = 20 # Initial cap, can be adjusted
        
        # Let's refine the strategy:
        # We have 300 calls.
        # We will use the starts in order.
        # For each start, we run a (1+1)-ES.
        # We stop the current (1+1)-ES if:
        #   - We hit stagnation_limit (no improvement for N steps)
        #   - We have used up the budget
        #   - Time limit is approaching
        
        for start_idx, start_point in enumerate(starts):
            if calls >= self.budget:
                break
            if time.time() > deadline - 10: # Stop 10s early
                break
                
            x = start_point.copy()
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            sigma = 2.0 # Initial step size
            steps_since_improvement = 0
            success_count = 0
            total_steps = 0
            
            # (1+1)-ES loop for this restart
            while calls < self.budget:
                if time.time() > deadline - 5:
                    break
                
                # Generate offspring
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    success_count += 1
                    steps_since_improvement = 0
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    # 1/5 rule: if success rate > 0.2, increase sigma
                    # We use a window or just simple ratio. 
                    # Simple approach: maintain a count.
                    # To keep it simple and fast, we can use a small window.
                    # Or just: if success, sigma *= 1.1, else sigma *= 0.9? 
                    # The 1/5 rule is more robust.
                    # Let's use a sliding window of last 20 steps.
                    # For simplicity in this tight loop, let's use a simplified adaptive rule:
                    # If we improved, sigma *= 1.05
                    # If not, sigma *= 0.95
                    # This is a bit slow to adapt. 
                    # Let's stick to the 1/5 rule with a small buffer.
                    pass 
                else:
                    steps_since_improvement += 1
                    # No improvement
                    pass
                
                # Adaptive Step Size (Simplified 1/5 rule)
                # We need to track successes in a window.
                # To avoid overhead, let's use a simple heuristic:
                # If steps_since_improvement == 0 (just improved), sigma *= 1.1
                # Else sigma *= 0.9
                # This is not exactly 1/5 but works well for (1+1)-ES.
                if steps_since_improvement == 0:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                # Ensure sigma doesn't get too small or too large
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                total_steps += 1
                
                # Check for stagnation
                if steps_since_improvement >= stagnation_limit:
                    break
                # Check for max steps per restart to force diversity
                if total_steps >= max_steps_per_restart:
                    break
            
            # If we broke due to stagnation or max steps, try next start.
            # If we broke due to budget/time, the outer loop will handle it.
            
        return best_x, best_f
