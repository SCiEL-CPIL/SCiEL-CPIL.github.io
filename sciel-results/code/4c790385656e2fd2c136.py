"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Two-phase hybrid: radial sweep to find Discus needle axis, then aggressive (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 1: Dense deterministic radial sweep to identify needle axis ---
        # Discus f11 has extreme anisotropy: one narrow "needle" direction.
        # We sweep along many deterministic directions from origin to find the best axis.
        
        # Generate deterministic directions: coordinate axes, diagonals, and golden-angle rotations
        directions = []
        
        # Coordinate axes (positive and negative)
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
            d[i] = -1.0
            directions.append(d)
        
        # Diagonals (all ±1 combinations, sampled)
        for mask in range(2**dim):
            if mask == 0:
                continue
            d = np.zeros(dim)
            for i in range(dim):
                if mask & (1 << i):
                    d[i] = 1.0
            norm = np.linalg.norm(d)
            if norm > 0:
                directions.append(d / norm)
        
        # Additional deterministic directions via golden angle rotation
        golden = np.pi * (3 - np.sqrt(5))
        for k in range(12):
            angle = k * golden
            d = np.zeros(dim)
            # Spread across dimensions using trigonometric functions
            for i in range(dim):
                d[i] = np.cos(angle * (i + 1) + i * 0.7)
            norm = np.linalg.norm(d)
            if norm > 0:
                directions.append(d / norm)
        
        # Remove duplicates (within tolerance)
        unique_dirs = []
        for d in directions:
            is_dup = False
            for u in unique_dirs:
                if abs(np.dot(d, u)) > 0.99:
                    is_dup = True
                    break
            if not is_dup:
                unique_dirs.append(d)
        
        directions = unique_dirs
        
        # Allocate budget: ~40% for phase 1 sweep, 60% for phase 2 ES
        sweep_budget = int(budget * 0.4)
        es_budget = budget - sweep_budget
        
        # For each direction, do a 1D sweep along that axis from origin
        # Find the best point along each direction, then track the global best
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1
        
        # For each direction, do a logarithmic sweep to find approximate minimum along that ray
        for d in directions:
            if calls >= sweep_budget:
                break
            
            # Sweep along direction d with logarithmic spacing
            # Try multiple radii to find where the function drops
            radii = [0.1, 0.3, 1.0, 2.0, 3.0, 4.0, 4.5]
            local_best_x = best_x.copy()
            local_best_f = best_f
            
            for r in radii:
                if calls >= sweep_budget:
                    break
                x = d * r
                x = np.clip(x, -5, 5)
                f = func(x)
                calls += 1
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x.copy()
            
            # If we found a better point, refine with a few more evaluations
            if local_best_f < best_f:
                best_x = local_best_x.copy()
                best_f = local_best_f
                
                # Quick refinement: try small perturbations around the best point
                if calls < sweep_budget:
                    for r in [0.5, 1.5, 2.5, 3.5]:
                        if calls >= sweep_budget:
                            break
                        x = d * r
                        x = np.clip(x, -5, 5)
                        f = func(x)
                        calls += 1
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
        
        # --- Phase 2: Aggressive (1+1)-ES with adaptive step size ---
        # Start from the best point found in phase 1
        x = best_x.copy()
        f = best_f
        
        # Initialize step size based on how far we are from origin
        # Use a relatively large initial step to exploit the needle structure
        dist_from_origin = np.linalg.norm(x)
        sigma = max(0.5, dist_from_origin * 0.3)
        
        # (1+1)-ES with success-rate-based step size adaptation
        # High mutation rate to quickly explore
        success_threshold = 0.35  # Target success rate
        c_sigma = 1.0 / (4 * dim)  # Fast adaptation
        h = 0.0  # Success history
        h_len = 10  # History length
        
        # For each iteration:
        # 1. Generate one mutation
        # 2. Evaluate
        # 3. Update step size based on success
        while calls < budget:
            # Generate mutation
            z = rng.standard_normal(dim)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls += 1
            
            # Update success history
            success = f_trial < f
            h = (1 - 1/h_len) * h + (1/h_len) * (1 if success else 0)
            
            # Adapt step size
            sigma *= np.exp(c_sigma * (h - success_threshold))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Accept if better
            if success:
                x = x_trial
                f = f_trial
        
        return x, f
# EVOLVE-BLOCK-END
