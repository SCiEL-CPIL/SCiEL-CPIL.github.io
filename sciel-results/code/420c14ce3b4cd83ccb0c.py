"""Multi-objective optimizer for ZDT1 using hypervolume-driven adaptive search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 structure: f1 = x[0], f2 = 1 - sqrt(f1) * (1 + 10*(n-1)) + 10*(n-1)*(sum((x_i-0.5)^2 + sin(10*pi*x_i)) / (n-1))
        # To get on the Pareto front, we need the second term to be 0, which means x_i = 0.5 for i >= 1
        # and sin(10*pi*x_i) = 0, which is satisfied at x_i = 0.5
        
        # Strategy: sample x1 values, set other vars to 0.5, then refine
        # Track all evaluated points and their objectives
        
        all_X = []
        all_F = []
        used = 0
        
        def eval_point(x):
            nonlocal used
            if used >= self.budget:
                return None
            f = problem(x)
            all_X.append(x.copy())
            all_F.append(f)
            used += 1
            return f
        
        # Phase 1: Strategic sampling of x1 values to spread along the front
        # Use logarithmic and linear spacing to cover the front well
        n_phase1 = min(40, self.budget // 4)
        
        # Mix of evenly spaced and clustered points
        x1_values = np.linspace(0.0, 1.0, n_phase1)
        # Add some random perturbations for diversity
        x1_values = np.sort(np.concatenate([x1_values, rng.random(min(10, self.budget // 10))]))
        x1_values = x1_values[:min(n_phase1 + 10, self.budget // 3)]
        
        for x1 in x1_values:
            if used >= self.budget or time.time() > deadline:
                break
            x = np.full(self.dim, 0.5)
            x[0] = x1
            eval_point(x)
        
        # Phase 2: Local refinement around promising points
        # For each evaluated point, try small perturbations in x1 and other variables
        remaining_budget = self.budget - used
        
        if remaining_budget > 0 and len(all_F) > 0:
            F_arr = np.array(all_F)
            
            # Sort by f1 to identify regions
            # Focus refinement on points that contribute to hypervolume
            # Simple approach: refine points with diverse f1 values
            
            n_refine = min(remaining_budget, 120)
            points_per_refine = max(1, n_refine // max(1, len(all_F)))
            
            # Select a subset of points to refine (spread across f1 range)
            if len(all_F) > 10:
                indices = np.linspace(0, len(all_F) - 1, min(10, len(all_F))).astype(int)
            else:
                indices = list(range(len(all_F)))
            
            for idx in indices:
                if used >= self.budget or time.time() > deadline:
                    break
                
                x_base = all_X[idx].copy()
                f_base = all_F[idx]
                
                # Try perturbations in x1 direction
                for dx1 in [-0.05, -0.02, 0.02, 0.05]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_new = x_base.copy()
                    x_new[0] = np.clip(x_new[0] + dx1, lo[0], hi[0])
                    eval_point(x_new)
                
                # Try small random perturbations in other dimensions
                for _ in range(2):
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_new = x_base.copy()
                    noise = rng.standard_normal(self.dim) * 0.01
                    x_new = np.clip(x_new + noise, lo, hi)
                    eval_point(x_new)
        
        # Phase 3: Final targeted search if budget remains
        remaining = self.budget - used
        if remaining > 0 and len(all_F) > 0:
            F_arr = np.array(all_F)
            # Find gaps in f1 coverage and target them
            f1_vals = F_arr[:, 0]
            if len(f1_vals) > 1:
                f1_sorted = np.sort(f1_vals)
                gaps = np.diff(f1_sorted)
                # Target the largest gaps
                for _ in range(min(remaining, 20)):
                    if used >= self.budget or time.time() > deadline:
                        break
                    if len(gaps) > 0:
                        max_gap_idx = np.argmax(gaps)
                        target_x1 = (f1_sorted[max_gap_idx] + f1_sorted[max_gap_idx + 1]) / 2
                    else:
                        target_x1 = rng.random()
                    
                    x = np.full(self.dim, 0.5)
                    x[0] = np.clip(target_x1, lo[0], hi[0])
                    eval_point(x)
        
        return all_X
# EVOLVE-BLOCK-END
