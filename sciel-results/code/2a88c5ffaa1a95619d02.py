"""Combined optimizer: global scan + multi-restart (1+1)-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global scan followed by multi-restart (1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Phase 1: Global scan to find promising basins
        n_scan = 30
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
        # We want to balance exploration vs exploitation.
        # With 500 budget, 30 used for scan, 470 remaining.
        # Let's use top 5 points.
        k = min(5, n_scan)
        top_idxs = sorted_idx[:k]
        
        best_x = xs[top_idxs[0]].copy()
        best_f = fs[top_idxs[0]]

        # Phase 2: Local refinement using (1+1)-ES on each top point
        # Distribute remaining budget among the top k points.
        # We can't know exactly how many calls each will use, so we just
        # run them sequentially until time or global budget is exhausted.
        # To ensure early progress is captured, we interleave or just
        # run the best one first? 
        # Running the best one first allows it to potentially find the global
        # minimum quickly. If it gets stuck, we move to the next.
        # However, if the first one is in a local minimum and takes 200 calls
        # to plateau, we lose time.
        # A better strategy: allocate a fixed portion of budget to each,
        # or run them in a round-robin fashion?
        # Given the (1+1)-ES nature, it converges. 
        # Let's run the top 1 for a portion, then top 2, etc.
        # Or simply: run the best one until it stagnates or budget runs out.
        # But if the best scan point is not the global minimum, we are stuck.
        # The scan of 30 points in 5D is sparse. The probability of hitting
        # the exact basin of the global minimum is low but non-zero.
        # Actually, Gallagher 101 has many peaks. The "minimum" is one of them.
        # A random point has a 1/101 chance of being in the global basin (roughly).
        # 30 points gives ~1 - (100/101)^30 ~ 26% chance of hitting the global basin.
        # So it's worth scanning.
        
        # Strategy:
        # For each of the top k points (sorted by best first):
        #   Run (1+1)-ES with a local budget limit.
        #   If we find a better point, update global best.
        
        # Let's allocate budget dynamically.
        # If we have 470 calls left, and 5 points, we could give ~94 calls to each.
        # But if the first one finds the optimum, we want to use all remaining.
        
        # Simpler: Just run the (1+1)-ES on the best point found so far.
        # But restarts are key for multi-modal.
        # Let's do:
        # 1. Run (1+1)-ES on top_idxs[0] with max 200 calls.
        # 2. Run (1+1)-ES on top_idxs[1] with max 150 calls.
        # 3. Run (1+1)-ES on top_idxs[2] with max 100 calls.
        # 4. Run (1+1)-ES on top_idxs[3] with max 50 calls.
        # 5. Run (1+1)-ES on top_idxs[4] with max 50 calls.
        # Total: 550 > 470. So we will hit the limit.
        
        # Better: Just loop through top points and run ES until time/budget up.
        # We'll track total calls.
        
        for idx in top_idxs:
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = xs[idx].copy()
            fx = fs[idx]
            
            # (1+1)-ES parameters
            sigma = 2.0
            # We want to spend some budget here.
            # Let's say we allow up to 100 calls per restart, or until time out.
            local_budget = 100
            no_improve_count = 0
            max_no_improve = 15
            
            for _ in range(local_budget):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.5
                    no_improve_count = 0
                else:
                    sigma *= 0.87
                    no_improve_count += 1
                    
                # Early stop if sigma becomes very small or no progress
                if sigma < 1e-6 or no_improve_count >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
