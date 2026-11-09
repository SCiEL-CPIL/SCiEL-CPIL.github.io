"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Global Scan + CMA-ES Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Global state
        global_best_x = np.zeros(dim)
        global_best_f = float('inf')
        calls = 0
        start_time = time.time()
        max_time = 280.0 # Safety margin for 330s limit

        def can_continue():
            return calls < budget and (time.time() - start_time) < max_time

        def update_best(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()

        # --- Phase 1: Global Scan ---
        # For dim=5, a grid of size 3^5 = 243 points is affordable and provides 
        # excellent coverage of [-5, 5]^5. This ensures we find the correct 
        # basin quickly, which is vital for AOCC.
        
        # We use a slightly denser grid if budget allows, but 3^5 is a good start.
        # Points: -5, 0, 5.
        # Actually, to be more effective for Gallagher 101, we might want to 
        # avoid the exact center if it's a peak, but we don't know the shift.
        # Let's use 4 points per dim? 4^5 = 1024 > 500. Too many.
        # 3^5 = 243. Remaining budget for CMA-ES: 500 - 243 = 257.
        
        # Let's generate the grid
        grid_vals = np.array([-4.0, 0.0, 4.0]) # Avoid boundaries slightly? 
        # The domain is [-5, 5]. Min is in [-4, 4]. 
        # Let's use [-4, 0, 4] to stay well inside.
        
        # Generate all combinations
        # Using meshgrid
        axes = [grid_vals] * dim
        grids = np.meshgrid(*axes, indexing='ij')
        # Flatten to (N, dim)
        scan_points = np.vstack([g.flatten() for g in grids])
        
        # Shuffle to ensure we don't waste time if one region is bad? 
        # Or just evaluate sequentially? 
        # For AOCC, finding a good point early is key. 
        # Randomizing the order of the scan might help if the best point is "late" in the grid.
        # Let's shuffle the scan points.
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(scan_points))
        scan_points = scan_points[perm]
        
        # Evaluate scan points
        # We can vectorize if func supports it, but the contract says "func(x) ... returns a float".
        # So we must loop.
        
        for x in scan_points:
            if not can_continue():
                break
            # Clip just in case, though grid is inside
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls += 1
            update_best(x_clipped, f_val)
            
        if not can_continue():
            return global_best_x, global_best_f

        # --- Phase 2: CMA-ES Refinement ---
        # Initialize CMA-ES at the best point found in the scan.
        # Use a large sigma to allow exploration of the local basin structure.
        # The basin of attraction for Gallagher 101 peaks can be small.
        # A sigma of 1.0 or 1.5 is reasonable to explore the neighborhood.
        
        x0 = global_best_x.copy()
        sigma0 = 1.5
        
        # CMA-ES Parameters
        lambda_pop = 10 # Fixed for dim=5
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        # State variables
        x_mean = x0
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute Cholesky/Eig for sampling
        # Initial: Identity
        B = np.eye(dim)
        D = np.ones(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim * dim))
        
        gen = 0
        while can_continue():
            # Generate population
            # Use a fresh RNG for each generation to ensure reproducibility if needed, 
            # or just continue the stream. Let's use a dedicated RNG.
            rng_gen = np.random.default_rng(1000 + gen)
            y = rng_gen.standard_normal((lambda_pop, dim))
            x_pop = x_mean + sigma * (y @ B.T * D)
            
            # Evaluate population
            f_pop = np.empty(lambda_pop)
            for i in range(lambda_pop):
                if not can_continue():
                    # If we run out of budget mid-generation, break
                    # We need to handle partial generations carefully or just stop.
                    # Standard CMA-ES assumes full populations. 
                    # If we break, we should probably not update CMA-ES state to avoid bias.
                    # Let's just stop the loop.
                    return global_best_x, global_best_f
                
                x_i = np.clip(x_pop[i], -5, 5)
                f_i = func(x_i)
                calls += 1
                f_pop[i] = f_i
                update_best(x_i, f_i)
            
            # Sort
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]
            f_sorted = f_pop[order]
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update covariance matrix
            diff = x_sorted[:mu] - x_mean
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - cc) * C) + cmu * (diff.T @ (weights[:, np.newaxis] * diff))
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update evolution paths
            # Need C^-1/2. 
            # For dim=5, eigendecomposition is fast.
            eigval, eigvec = np.linalg.eigh(C)
            # Handle numerical issues
            eigval = np.maximum(eigval, 1e-12)
            D = np.sqrt(eigval)
            B = eigvec
            Csqrt_inv = B / D
            
            pc = (1 - cc) * pc + cc * np.sqrt(mueff) / (sigma * D) @ (x_new - x_mean)
            
            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Csqrt_inv @ (x_new - x_mean) / sigma)
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x_mean = x_new
            
            # Termination check
            if sigma < 1e-12:
                break
            if np.linalg.norm(x_new - x_mean) < 1e-12 * (1 + np.linalg.norm(x_mean)):
                 # No movement
                 pass
            
            gen += 1
            
            # Optional: If we have made very little progress for many generations, 
            # we might want to restart or just let it run. 
            # Given the tight budget, we just run until budget is exhausted.
            
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
