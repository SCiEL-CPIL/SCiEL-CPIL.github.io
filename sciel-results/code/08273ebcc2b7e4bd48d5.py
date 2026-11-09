import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, return before 330s wall clock kill. 
        # Using 280s deadline to be safe against overhead.
        deadline = start_time + 280.0
        
        rng = np.random.default_rng()
        
        best_x = None
        best_f = float('inf')
        
        calls_used = 0
        max_calls = self.budget
        
        # Strategy: Two-phase approach
        # Phase 1: Global Exploration (Multi-Start Initialization)
        # Phase 2: Local Exploitation (Adaptive Step Size Hill Climbing / CMA-ES lite)
        
        # Phase 1: Use 10% of budget for global search to find a good starting point
        # 500 * 0.1 = 50 calls.
        # We will sample 50 random points.
        n_global = 50
        
        # Sample points
        # To ensure coverage, we can use a mix of uniform random and maybe some structured sampling
        # but uniform random is standard for BBOB.
        
        for i in range(n_global):
            if time.time() > deadline or calls_used >= max_calls:
                break
            
            x = rng.uniform(-5, 5, size=self.dim)
            fx = func(x)
            calls_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if we didn't even make one call (shouldn't happen)
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_used += 1
        
        # Phase 2: Local Search
        # We use an adaptive step size hill climber.
        # This is effectively a (1+1) ES with step size adaptation.
        # We track the step size sigma.
        
        # Initialize sigma based on the box size and dimension
        # A good initial sigma is roughly the box width / sqrt(dim) or similar.
        # Box width is 10. Dim is 5.
        # Let's start with a relatively large step to escape shallow local minima if the global search was lucky but not perfect.
        sigma = 1.0 
        
        # We will iterate until budget or time is exhausted.
        # We need to balance exploration (large sigma) and exploitation (small sigma).
        
        # Adaptation rule:
        # If we improve, keep sigma or shrink slightly (0.9).
        # If we don't improve, expand sigma (1.1) to try to jump out of local minima.
        # However, pure (1+1) ES can get stuck. 
        # To improve robustness, we can occasionally "restart" the local search from a slightly perturbed best_x 
        # if we stagnate for a long time.
        
        # Let's implement a stagnation counter.
        stagnation_count = 0
        max_stagnation = 20 # If no improvement for 20 steps, increase sigma more aggressively or restart
        
        while calls_used < max_calls and time.time() < deadline:
            # Generate a candidate
            # We try to move in a random direction
            direction = rng.normal(0, 1, size=self.dim)
            direction /= np.linalg.norm(direction) # Normalize to unit vector
            
            # Step size is sigma
            x_new = best_x + sigma * direction
            x_new = np.clip(x_new, -5, 5)
            
            fx_new = func(x_new)
            calls_used += 1
            
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
                stagnation_count = 0
                # Shrink step size to fine-tune
                sigma *= 0.95
            else:
                stagnation_count += 1
                # Expand step size to explore more
                # If stagnation is high, expand more aggressively
                if stagnation_count > max_stagnation:
                    sigma *= 1.2
                    # Optional: If stagnation is very high, consider a larger jump
                    # But let's stick to simple adaptation first.
                else:
                    sigma *= 1.05
            
            # Clamp sigma to reasonable bounds
            # Min sigma: very small, to avoid numerical noise
            # Max sigma: large enough to cross the box
            sigma = np.clip(sigma, 1e-6, 10.0)
        
        return best_x, best_f
