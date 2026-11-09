"""Combined optimizer: global scan + multi-restart (1+1)-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global scan followed by multi-restart (1+1)-ES with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
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
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            return x, fx

        # Sort scan results
        sorted_idx = np.argsort(fs)
        
        # Select top K points for local refinement
        k = min(5, n_scan)
        top_idxs = sorted_idx[:k]
        
        best_x = xs[top_idxs[0]].copy()
        best_f = fs[top_idxs[0]]

        # Phase 2: Local refinement using (1+1)-ES on each top point
        # Use larger initial sigma for broader exploration of basins
        # Fixed budget per restart to ensure fair allocation
        local_budget_per_restart = 90  # 5 restarts * 90 = 450, leaving buffer
        
        for idx in top_idxs:
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = xs[idx].copy()
            fx = fs[idx]
            
            # (1+1)-ES with parameters tuned for Gallagher 101
            sigma = 2.0
            no_improve = 0
            max_no_improve = 20
            
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
                    sigma *= 0.87
                    no_improve += 1
                    
                # Early stop if sigma becomes very small or stagnation detected
                if sigma < 1e-6 or no_improve >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
