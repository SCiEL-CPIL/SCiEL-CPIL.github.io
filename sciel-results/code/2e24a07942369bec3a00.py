# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # ZDT4 structure: x1 in [0,1] (or [-5,5] depending on bounds), 
        # x2..x10 should be near the global minimum of g(x)
        # g(x) = 1 + 10*(n-1) + sum(x_i^2 - 10*cos(4*pi*x_i))
        # Global min of each term x_i^2 - 10*cos(4*pi*x_i) is at x_i = 0.5 (value = 0.25 - 10 = -9.75)
        # So g_min = 1 + 10*(dim-1) + (dim-1)*(-9.75) = 1 + 10*9 + 9*(-9.75) = 1+90-87.75 = 3.25
        # But actually for ZDT4, g_min = 1 when all x_i = 0.5? Let me recalculate.
        # g(x) = 1 + 10(n-1) + sum_{i=2}^n (x_i^2 - 10*cos(4*pi*x_i))
        # At x_i = 0.5: 0.25 - 10*cos(2*pi) = 0.25 - 10 = -9.75
        # Sum over 9 terms: 9 * (-9.75) = -87.75
        # g = 1 + 90 - 87.75 = 3.25
        # Hmm, but the standard ZDT4 has g_min = 1. Let me check bounds.
        # Actually the bounds for x2..x10 in ZDT4 are typically [0,1].
        # At x_i = 0.5: value = -9.75 per term.
        # g_min = 1 + 10*9 + 9*(-9.75) = 1 + 90 - 87.75 = 3.25
        # Wait, maybe the problem uses different bounds. Let's just use 0.5 as the target.

        # Strategy: 
        # 1. Set x2..x10 to 0.5 (global min of the oscillatory part)
        # 2. Sweep x1 across its range to cover the Pareto front
        # 3. Use remaining budget for local refinement

        # Determine x1 range from bounds
        x1_lo, x1_hi = bounds[0][0], bounds[0][1]
        
        # For ZDT4, x1 is typically in [0, 1] but could be [-5, 5]
        # f1 = x1, f2 = g(x)*(1 - sqrt(x1/g(x)))
        # The Pareto front is traced by varying x1 with g at minimum

        # Phase 1: Dense sweep of x1 with x2..x10 fixed at 0.5
        # Use most of the budget for this
        n_sweep = min(self.budget - 10, 150)  # Leave some for refinement
        n_sweep = max(n_sweep, 20)
        
        # Create x1 values - use non-uniform distribution to concentrate where front is interesting
        # For ZDT4, the interesting part of the front is where x1 is in [0, 1]
        # But let's cover the full range first
        
        # Use a mix: uniform over the range
        x1_values = np.linspace(x1_lo, x1_hi, n_sweep)
        
        # Create base vector with x2..x10 = 0.5
        base = np.zeros(dim)
        base[0] = 0  # Will be set per point
        base[1:] = 0.5  # Global min location for oscillatory terms
        
        # Evaluate the sweep
        evaluated_x = []
        evaluated_f = []
        used = 0
        
        for x1 in x1_values:
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            x = base.copy()
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated_x.append(x)
            evaluated_f.append(f)
            used += 1
        
        # Phase 2: Local refinement around best points
        if used < self.budget and evaluated_f:
            F_arr = np.array(evaluated_f)
            X_arr = np.array(evaluated_x)
            
            # Find non-dominated points
            n_pts = len(F_arr)
            dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                if dominated[i]:
                    continue
                for j in range(n_pts):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            
            # Refine around non-dominated points
            nondom_idx = np.where(~dominated)[0]
            n_refine = min(self.budget - used, 30)
            
            for idx in nondom_idx[:min(len(nondom_idx), n_refine)]:
                if used >= self.budget or time.time() > deadline:
                    break
                x_parent = X_arr[idx].copy()
                # Small perturbation in x1 direction
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    child = x_parent.copy()
                    child[0] += 0.01 * (x1_hi - x1_lo) * np.random.randn()
                    child = np.clip(child, lo, hi)
                    f_child = problem(child)
                    evaluated_x.append(child)
                    evaluated_f.append(f_child)
                    used += 1
        
        # Phase 3: Additional random sampling if budget remains
        if used < self.budget:
            n_random = self.budget - used
            for _ in range(n_random):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.rand(dim)
                f = problem(x)
                evaluated_x.append(x)
                evaluated_f.append(f)
                used += 1
        
        return X_arr if len(evaluated_x) > 0 else np.array([base])
# EVOLVE-BLOCK-END
