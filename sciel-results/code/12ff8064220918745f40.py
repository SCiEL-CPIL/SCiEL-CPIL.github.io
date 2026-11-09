"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Learning rates
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Step size
        sigma = 2.0  # Initial sigma, box is [-5, 5]^5, center 0
        # Mean (initial position)
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        # Evolution path
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Initial evaluation
        calls_made = 0
        best_fx = float('inf')
        best_x = mean.copy()
        
        # Evaluate initial mean
        fx_mean = func(mean)
        calls_made += 1
        if fx_mean < best_fx:
            best_fx = fx_mean
            best_x = mean.copy()
            
        # Pre-compute D (sqrt of C) initially
        D = np.sqrt(np.linalg.eigvalsh(C))
        
        # Main Loop
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate population
            # We need to generate lambda_ points. 
            # To save time, we can batch generate if possible, but func is black box.
            # We must call func sequentially to respect budget and time checks.
            
            ys = []
            fvals = []
            
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Standard normal
                z = np.random.randn(dim)
                
                # Transform
                # y = mean + sigma * D @ (C^{-1/2} @ z) ... wait, standard CMA-ES uses
                # y = mean + sigma * (C^{1/2} @ z)
                # Efficient update:
                # We maintain C. We need C^{1/2}.
                # Recomputing eigendecomp every step is expensive (O(n^3)).
                # For dim=5, it's cheap.
                
                # Eigendecomposition
                # Note: For small dim, this is fast enough.
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C_sqrt = np.eye(dim)
                
                y = mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                ys.append(y)
                fvals.append(fy)
                
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            
            if len(ys) < lambda_:
                break
                
            ys = np.array(ys)
            fvals = np.array(fvals)
            
            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = ys[order]
            fvals_sorted = fvals[order]
            
            # Update mean
            mean_new = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            
            # Update evolution paths
            # Need C^{-1/2} for ps
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(dim)
                
            # pc = (1-cc)pc + cc/sigma * (mean_new - mean)
            pc = (1 - cc) * pc + cc / sigma * (mean_new - mean)
            
            # ps = (1-cs)ps + sqrt(cs(2-cs)(1-cc^2)) / sigma * C_inv_sqrt @ (mean_new - mean)
            hsig = 1.0
            # Heuristic for hsig
            norm_pc = np.linalg.norm(pc)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2) - 1 / (8 * dim**4))
            if norm_pc / np.sqrt(1 - (1 - cc) ** (2 * (calls_made // lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n:
                hsig = 1.0
            else:
                hsig = 0.0
                
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * (1 - cc ** 2)) / sigma * (C_inv_sqrt @ (mean_new - mean))
            
            # Update Covariance
            # Rank-mu update
            v = (mean_new - mean) / sigma
            rank_mu = np.sum(weights[:, None] * (ys_sorted[:mu] - mean)[:, None] * (ys_sorted[:mu] - mean)[:, None], axis=0) / sigma**2
            
            # Rank-one update
            rank_one = np.outer(pc, pc)
            
            # Correction term
            delta_hs = (1 - hsig) * cs * (2 - cs) * np.outer(ps, ps)
            
            C = (1 + c1 * (2 - c1 - cmu) + cmu) * C - c1 * rank_one - cmu * rank_mu + delta_hs
            
            # Ensure positive definite (numerical stability)
            # Add small identity if needed
            C += 1e-12 * np.eye(dim)
            
            mean = mean_new
            
            # Adapt sigma
            norm_ps = np.linalg.norm(ps)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2) - 1 / (8 * dim**4))
            sigma *= np.exp(cs / (dim + 3) * (norm_ps / chi_n - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Check convergence
            if np.linalg.norm(mean_new - mean) < 1e-6:
                 # If stuck, maybe restart or break? 
                 # For simplicity, break if no progress for a while? 
                 # CMA-ES handles this via sigma adaptation.
                 pass

        return best_x, best_fx

# EVOLVE-BLOCK-END
