"""Combined optimizer: adaptive global scan + multi-restart (1+1)-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive global scan followed by multi-restart (1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Phase 1: Global scan to find promising basins
        # We use a larger initial scan (50 points) to increase the chance of hitting
        # the global basin in a multi-modal landscape like Gallagher 101.
        # 50 calls out of 500 is 10% of the budget, which is acceptable for exploration.
        n_scan = 50
        xs = rng.uniform(-5, 5, size=(n_scan, self.dim))
        fs = np.full(n_scan, np.inf)
        
        calls_made = 0
        for i in range(n_scan):
            if time.time() > deadline:
                break
            fx = func(xs[i])
            fs[i] = fx
            calls_made += 1
        
        if calls_made == 0:
            # Fallback if even scan fails (shouldn't happen)
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            return x, fx

        # Sort scan results
        sorted_idx = np.argsort(fs)
        
        # Select top K points for local refinement
        # With 500 budget, 50 used for scan, 450 remaining.
        # We select top 5 distinct points.
        k = min(5, n_scan)
        top_idxs = sorted_idx[:k]
        
        best_x = xs[top_idxs[0]].copy()
        best_f = fs[top_idxs[0]]

        # Phase 2: Local refinement using (1+1)-ES on each top point
        # We distribute the remaining budget among the top k points.
        # To handle the multi-modality, we restart from each of the top k points.
        # We allocate a portion of the remaining budget to each restart.
        # If a restart finds a significantly better point, we continue refining it.
        # If it stagnates, we move to the next restart.
        
        # Strategy:
        # For each of the top k points:
        #   Run (1+1)-ES for a fixed number of steps or until stagnation.
        #   Update global best.
        
        # Remaining budget
        remaining_budget = self.budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f
        
        # We want to ensure we don't spend too much on a single local minimum.
        # Let's allocate roughly remaining_budget / k to each, but with a cap.
        # Since (1+1)-ES can converge quickly, we can also stop early if sigma is tiny.
        
        # Let's try a dynamic allocation:
        # Start with a reasonable chunk, say 20% of remaining, for the first one.
        # But simpler is to just loop through them and let the global budget cap it.
        # However, to ensure we visit all top candidates, we should limit each.
        
        # Let's give each restart a maximum of (remaining_budget // k) + 10 calls.
        # But if one is very good, we might want to spend more.
        # Actually, for AOCC, early progress is key.
        # The scan already gives us some progress.
        # The ES should refine quickly.
        
        # Let's just run them sequentially, each with a limit of remaining_budget // k.
        # If budget runs out, we stop.
        
        budget_per_restart = remaining_budget // k
        if budget_per_restart < 10:
            budget_per_restart = 10 # Minimum effort per restart
            
        for idx in top_idxs:
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = xs[idx].copy()
            fx = fs[idx]
            
            # (1+1)-ES parameters
            sigma = 2.0
            
            # Run ES for this restart
            local_steps = 0
            max_local_steps = budget_per_restart
            
            while local_steps < max_local_steps:
                if calls_made >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                local_steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                    
                # Optional: early stop if sigma becomes very small
                if sigma < 1e-6:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
