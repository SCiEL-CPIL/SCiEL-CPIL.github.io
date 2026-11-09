import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 300s limit, return before 330s wall clock kill
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        # Use a fixed seed for reproducibility in evaluation if needed, 
        # but random is usually better for exploration diversity across instances.
        # However, the task says "hidden seeded shift", implying the function itself is fixed per instance.
        # We should use a different seed or default rng to avoid systematic bias if possible, 
        # but standard practice in these black-box tasks is often to just use np.random.
        
        rng = np.random.default_rng()
        
        best_x = None
        best_f = float('inf')
        
        calls_used = 0
        max_calls = self.budget
        
        # Strategy: Multi-start Random Hill Climbing (RHC) with adaptive step size.
        # RHC is simple and robust. We restart when no improvement is seen for a while.
        # The key for AOCC is to make progress early. 
        # A pure random search is bad. A pure local search gets stuck.
        # A "simulated annealing" like approach or "stochastic hill climbing" works well.
        
        # Let's use a simple "Best-Improvement" or "First-Improvement" RHC.
        # Since we have 500 calls, we can afford many small steps.
        
        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        calls_used += 1
        best_x = x.copy()
        best_f = fx
        
        # Adaptive step size
        # Start with a large step to explore, shrink as we get closer to optimum?
        # Or keep it constant? 
        # For f22, the peaks are spread out. A step of ~1.0 is reasonable.
        # Let's try a strategy where we do a few steps, then if stuck, restart from best_x with a new random direction.
        
        step_size = 1.0
        
        # We will iterate until budget is exhausted or time limit reached.
        # To maximize AOCC, we want to update best_f frequently.
        
        while calls_used < max_calls and time.time() < deadline:
            # Generate a candidate
            # Perturb current best? Or current point?
            # Standard RHC: perturb current point. If better, accept. Else, reject.
            # But if we reject, we are stuck. 
            # Better: "Greedy" RHC: Always perturb the BEST point found so far.
            # This ensures we are always moving towards the current best, but might get stuck in its basin.
            # To escape, we occasionally jump from the best point to a random location.
            
            # Hybrid approach:
            # 80% chance: Local move around best_x
            # 20% chance: Global jump (restart)
            
            if rng.random() < 0.2:
                # Global restart
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Local move around best_x
                # Use Gaussian perturbation
                noise = rng.normal(0, step_size, self.dim)
                x_new = best_x + noise
                # Clip to box
                x_new = np.clip(x_new, -5, 5)
                
                fx_new = func(x_new)
                calls_used += 1
                
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
                    # Optional: shrink step size slightly on improvement?
                    # step_size *= 0.95
                else:
                    # No improvement. 
                    # If we are stuck for a while, we might want to increase step size or force a restart.
                    # For simplicity, let the 20% global restart handle it.
                    pass
            
            # Check time frequently to avoid overshooting
            if time.time() > deadline - 5.0:
                break

        return best_x, best_f
