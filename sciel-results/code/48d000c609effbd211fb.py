"""Combined optimizer: stratified global scan + multi-restart (1+1)-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Stratified global scan followed by multi-restart (1+1)-ES with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Phase 1: Global scan
        # Use 30 random points to allow more budget for local refinement
        # This aligns with high-performing inspiration
        n_scan = 30
        xs = rng.uniform(-5, 5, (n_scan, self.dim))
        
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
        k = min(4, n_scan)
        top_idxs = sorted_idx[:k]
        
        best_x = xs[top_idxs[0]].copy()
        best_f = fs[top_idxs[0]]

        # Phase 2: Local refinement using (1+1)-ES on each top point
        # Use larger initial sigma and faster growth to escape broad minima
        # 4 restarts * ~117 = 468, total 30+468=498 < 500
        local_budget_per_restart = 117
        
        for idx in top_idxs:
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = xs[idx].copy()
            fx = fs[idx]
            
            # (1+1)-ES with parameters tuned for Gallagher 101
            # Increased growth factor to 1.5 based on inspiration
            sigma = 2.0
            no_improve = 0
            max_no_improve = 25
            
            for _ in range(local_budget_per_restart):
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
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1
                    
                # Early stop if sigma becomes very small or stagnation detected
                if sigma < 1e-5 or no_improve >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
