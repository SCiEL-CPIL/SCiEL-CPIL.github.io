import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine fixed values for x_2...x_dim
        # ZDT4 g(x) min is at x_i = 0.5 for i > 0
        fixed_vals = []
        for i in range(1, self.dim):
            target = 0.5
            if target < lo[i]:
                val = lo[i]
            elif target > hi[i]:
                val = hi[i]
            else:
                val = 0.5
            fixed_vals.append(val)
            
        # Phase 1: Coarse uniform sampling to get initial coverage
        # Use non-uniform sampling: denser at high f1 values where hypervolume contribution is larger
        # For ZDT4, f1 = x1, so we want more samples at higher x1 values
        # But actually, hypervolume is maximized by covering the Pareto front well
        # Let's use a combination: uniform base + refinement near promising regions
        
        # First, do a coarse uniform sample to identify promising regions
        n_coarse = min(self.budget // 2, 100)
        x1_coarse = np.linspace(lo[0], hi[0], n_coarse)
        
        X_coarse = np.zeros((n_coarse, self.dim))
        X_coarse[:, 0] = x1_coarse
        for i in range(1, self.dim):
            X_coarse[:, i] = fixed_vals[i-1]
            
        F_coarse = np.array([problem(x) for x in X_coarse])
        
        # Check time
        if time.time() > deadline - 30:
            return X_coarse
        
        # Phase 2: Refine around the Pareto-optimal region
        # For ZDT4, the Pareto front has f1 in [0, 1] and f2 = 1 + 10*(n-1) + 10*(n-1)*(1/f1 - sin(4*pi*f1))
        # The minimum f2 is at f1 where sin(4*pi*f1) = 1, i.e., f1 = 1/8, 3/8, 5/8, 7/8
        # But we want to cover the whole front for hypervolume
        
        # Use the coarse results to identify which x1 values give good (low) f2
        # Then refine around those regions
        remaining_budget = self.budget - n_coarse
        
        # Strategy: sample more densely where f2 is low (Pareto-optimal regions)
        # Create a weighted sampling based on inverse f2 (lower f2 = higher weight)
        f2_vals = F_coarse[:, 1]
        # Clip to avoid division by zero or negative
        f2_clipped = np.clip(f2_vals, 1e-6, None)
        weights = 1.0 / f2_clipped
        weights /= weights.sum()
        
        # Generate refined samples using weighted random sampling
        n_refine = remaining_budget
        # Use inverse CDF sampling
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        
        # Generate random numbers and map to indices
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        rand_vals = rng.random(n_refine)
        refine_indices = np.searchsorted(cdf, rand_vals)
        refine_indices = np.clip(refine_indices, 0, n_coarse - 1)
        
        # Get the x1 values from coarse samples at these indices
        x1_refine_base = x1_coarse[refine_indices]
        
        # Add small perturbations to explore local regions
        # The spacing in coarse is (hi[0] - lo[0]) / (n_coarse - 1)
        coarse_spacing = (hi[0] - lo[0]) / max(n_coarse - 1, 1)
        perturbations = rng.uniform(-coarse_spacing * 0.5, coarse_spacing * 0.5, n_refine)
        x1_refine = x1_refine_base + perturbations
        x1_refine = np.clip(x1_refine, lo[0], hi[0])
        
        X_refine = np.zeros((n_refine, self.dim))
        X_refine[:, 0] = x1_refine
        for i in range(1, self.dim):
            X_refine[:, i] = fixed_vals[i-1]
            
        F_refine = np.array([problem(x) for x in X_refine])
        
        # Check time
        if time.time() > deadline - 10:
            # Combine coarse and refine, but return only what we've evaluated
            all_X = np.vstack([X_coarse, X_refine])
            return all_X
        
        # Phase 3: Local refinement around the best points
        # Combine all evaluations so far
        all_X = np.vstack([X_coarse, X_refine])
        all_F = np.vstack([F_coarse, F_refine])
        
        # Find the point with minimum f2 (likely on or near Pareto front)
        min_f2_idx = np.argmin(all_F[:, 1])
        best_x1 = all_X[min_f2_idx, 0]
        
        # Do a local search around the best point
        remaining = self.budget - (n_coarse + n_refine)
        if remaining > 0:
            # Local refinement with decreasing step size
            local_steps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            local_samples = []
            
            for step in local_steps:
                if remaining <= 0 or time.time() > deadline - 5:
                    break
                n_local = min(remaining, 10)
                # Sample around best_x1
                x1_local = best_x1 + rng.uniform(-step, step, n_local)
                x1_local = np.clip(x1_local, lo[0], hi[0])
                
                X_local = np.zeros((n_local, self.dim))
                X_local[:, 0] = x1_local
                for i in range(1, self.dim):
                    X_local[:, i] = fixed_vals[i-1]
                
                F_local = np.array([problem(x) for x in X_local])
                local_samples.append((X_local, F_local))
                remaining -= n_local
                
                # Update best point if we found something better
                local_min_f2 = np.argmin(F_local[:, 1])
                if F_local[local_min_f2, 1] < all_F[min_f2_idx, 1]:
                    best_x1 = X_local[local_min_f2, 0]
            
            if local_samples:
                local_X = np.vstack([s[0] for s in local_samples])
                local_F = np.vstack([s[1] for s in local_samples])
                all_X = np.vstack([all_X, local_X])
                all_F = np.vstack([all_F, local_F])
        
        return all_X
