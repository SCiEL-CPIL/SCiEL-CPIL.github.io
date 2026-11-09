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

        # Phase 1: Stratified global scan to find promising basins
        # Use stratified sampling for better coverage of [-5, 5]^5
        # 2^5 = 32 cells, pick one random point per cell
        n_per_dim = 2
        # Create stratified grid
        grid_coords = []
        for i in range(n_per_dim):
            low = -5 + i * 10 / n_per_dim
            high = -5 + (i + 1) * 10 / n_per_dim
            grid_coords.append((low, high))
        
        # Generate one point per cell
        n_scan = n_per_dim ** self.dim  # 32 points
        xs = np.empty((n_scan, self.dim))
        idx = 0
        for i1 in range(n_per_dim):
            for i2 in range(n_per_dim):
                for i3 in range(n_per_dim):
                    for i4 in range(n_per_dim):
                        for i5 in range(n_per_dim):
                            xs[idx] = [
                                rng.uniform(grid_coords[i1][0], grid_coords[i1][1]),
                                rng.uniform(grid_coords[i2][0], grid_coords[i2][1]),
                                rng.uniform(grid_coords[i3][0], grid_coords[i3][1]),
                                rng.uniform(grid_coords[i4][0], grid_coords[i4][1]),
                                rng.uniform(grid_coords[i5][0], grid_coords[i5][1]),
                            ]
                            idx += 1
        
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
        # Use 5 top points to balance exploration vs exploitation
        k = min(5, n_scan)
        top_idxs = sorted_idx[:k]
        
        best_x = xs[top_idxs[0]].copy()
        best_f = fs[top_idxs[0]]

        # Phase 2: Local refinement using (1+1)-ES on each top point
        # Use larger initial sigma for broader exploration of basins
        # Fixed budget per restart to ensure fair allocation
        # 5 restarts * 90 = 450, leaving buffer for scan (32 calls) -> total ~482 < 500
        local_budget_per_restart = 90
        
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
                    x, fy = y, fy
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
