"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Multi-Start CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        # Total budget is 500. We use 5 restarts of 100 evaluations each.
        # This is a structural change from a single long run to multiple short runs.
        # CMA-ES is generally more robust to local optima than (1+1)-ES, 
        # especially with restarts.
        
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts  # 100
        
        best_x = None
        best_f = np.inf
        
        # Use distinct seeds for each restart to ensure diversity
        # Seeds are chosen to be distinct but deterministic
        seeds = [42, 123, 456, 789, 1011]
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            rng = np.random.default_rng(seeds[r])
            
            # Initialize CMA-ES state
            # Start points are varied slightly to ensure different basins
            # are explored, but centered around 0 or random within bounds
            # For BBOB f6 (Attractive Sector), the optimum is shifted.
            # A good initialization is often near the center or random.
            # Let's use random initialization within [-2, 2] to be safe but not too wide.
            x0 = rng.uniform(-2, 2, self.dim)
            
            # Initial sigma: 1.0 is standard for CMA-ES on [-5,5] box
            sigma = 1.0
            x = x0.copy()
            
            # Covariance matrix C (identity)
            C = np.eye(self.dim)
            
            # Evolution paths
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            
            # Parameters for dim=5
            # Standard CMA-ES parameters
            mu = self.dim // 2  # 2 (for dim 5, mu=2 is common, or floor(dim/2))
            # Actually, for small dim, mu = dim//2 is 2.
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1 / np.sum(w**2)
            
            c_sigma = (mueff + 2) / (self.dim + mueff + 5)
            c_c = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
            c_1 = 2 / ((self.dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2)**2 + mueff))
            
            chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
            
            # Initial function evaluation
            fx = func(x)
            
            # Local best for this restart
            local_best_x = x.copy()
            local_best_f = fx
            
            # If this is the first restart, update global best
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
            
            # CMA-ES Loop
            for step in range(budget_per_restart - 1):
                if time.time() > deadline:
                    break
                
                # Sample lambda = 2*mu candidates
                lam = 2 * mu
                ys = rng.standard_normal((lam, self.dim))
                
                # Transform to search space
                # We need the square root of C. 
                # For efficiency and simplicity in this small dim, we can use
                # the fact that C is symmetric positive definite.
                # eigvalsh/eigh is O(n^3). For n=5, it's fast.
                # Alternatively, we can maintain the square root matrix or use
                # the standard CMA-ES update which works with C directly if we
                # compute the Mahalanobis distance correctly.
                # Standard CMA-ES uses: z = C^{1/2} y
                # Then x = x + sigma * z
                
                # Compute square root of C
                # Since C is updated by rank-1/rank-mu updates, it stays SPD.
                # Using eigh is robust.
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip negative eigenvalues due to numerical errors
                eigvals = np.clip(eigvals, 1e-10, None)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                
                zs = ys @ sqrt_C.T
                ys_new = x + sigma * zs
                ys_new = np.clip(ys_new, -5, 5)
                
                # Evaluate
                fs = np.array([func(y) for y in ys_new])
                
                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = ys_new[order]
                fs_sorted = fs[order]
                zs_sorted = zs[order]
                
                # Update best
                if fs_sorted[0] < local_best_f:
                    local_best_f = fs_sorted[0]
                    local_best_x = ys_sorted[0].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                
                # Weighted mean
                m = np.sum(w[:, np.newaxis] * ys_sorted[:mu], axis=0)
                
                # Update center
                x = m
                
                # Update covariance matrix
                # Save old C for update
                C_old = C.copy()
                
                # Update evolution path ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                    1 / sigma) * (zs_sorted[:mu] @ w)
                
                # Update evolution path pc
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (step + 1)))) < (1 + 2 / self.dim) * chiN
                hsig = 1 if hsig else 0
                
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (
                    1 / sigma) * (zs_sorted[:mu] @ w)
                
                # Update C
                # Term 1: (1 - c_1 - c_mu) * C
                # Term 2: c_1 * (pc @ pc.T + (1 - (1 - c_sigma)^2) * C_old)
                # Term 3: c_mu * sum(w_i * (y_i - m) @ (y_i - m).T)
                
                C = (1 - c_1 - c_mu) * C_old
                C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma)**2) * C_old)
                
                # Rank-mu update
                y_mu = (ys_sorted[:mu] - m) / sigma
                C += c_mu * np.sum(w[:, np.newaxis, np.newaxis] * (y_mu[:, np.newaxis, :] * y_mu[:, np.newaxis, :].T), axis=0)
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(c_sigma / chiN * (np.linalg.norm(ps) / chiN - 1))
                
                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f

# EVOLVE-BLOCK-END
