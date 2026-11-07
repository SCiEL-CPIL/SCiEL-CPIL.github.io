import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        t_start = time.time()
        deadline = t_start + 290.0
        
        # CMA-ES parameters
        # Population size
        lam = 4 if dim <= 5 else 6 + 3 * int(np.log(dim))
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Step size control
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        
        # Covariance matrix adaptation
        c1 = (2 * (mueff - 2 + 1 / mueff)) / ((dim + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initialization
        rng = np.random.default_rng(0)
        mean = np.zeros(dim)
        # Start with a slightly larger sigma to cover the box [-5, 5]
        sigma = 2.0 
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        best_x = mean.copy()
        best_f = float('inf')
        n_calls = 0
        
        # Precompute Cholesky factor of C
        # We will update B and D which is equivalent to Cholesky of C
        # C = B * diag(D^2) * B.T
        
        def cholesky_update(C, B, D):
            # Standard CMA-ES update for B and D
            # This is a simplified version. For dim=5, we can just recompute Cholesky if needed
            # or use the standard update.
            # For robustness and simplicity in this constrained rewrite, 
            # let's stick to the standard update formulas.
            pass

        # Main Loop
        while n_calls < budget:
            if time.time() > deadline:
                break
            
            # Generate samples
            samples = np.zeros((lam, dim))
            for i in range(lam):
                # Standard normal
                z = rng.standard_normal(dim)
                # Transform to N(0, C) using B and D
                # y = B * (D * z)
                y = B @ (D * z)
                samples[i] = mean + sigma * y
            
            # Clip to box
            samples = np.clip(samples, -5, 5)
            
            # Evaluate
            fvals = np.array([func(x) for x in samples])
            n_calls += lam
            
            # Track best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = samples[idx_best].copy()
                
            # Sort
            order = np.argsort(fvals)
            y_sorted = samples[order]
            f_sorted = fvals[order]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # Update step size
            # Evolution path for step size
            # Need Cholesky factor of C^-1/2
            # For dim=5, we can compute it on the fly or maintain it.
            # Let's compute C_inv_sqrt using eigen decomposition or Cholesky of C.
            # C = B @ (D**2) @ B.T
            # C_inv_sqrt = B @ (1/D) @ B.T
            C_inv_sqrt = B @ (1.0 / D) @ B.T
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (mean - old_mean) / sigma)
            
            # Update covariance adaptation path
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff * (1 - (1 - cs)**2)) * (mean - old_mean) / sigma
            # Note: The factor for pc update in standard CMA-ES is sqrt(cs * (2-cs) * mueff)
            # The condition for rank-mu update involves |pc|^2
            
            # Check for step size increase
            hsig = (np.sqrt((1 - (1 - cs)**(2 * (n_calls + 1)))) / np.sqrt((1 - (1 - cs)**2)) - 1) * np.sqrt(dim + 2) / np.sqrt(mueff)
            hsig_val = 1.0 if np.sqrt((np.sum(ps**2) / (1 - (1 - cs)**(2 * (n_calls + 1)))) / dim) < (1.4 + 2 / (dim + 1)) * hsig else 0.0
            
            sigma *= np.exp(cs / ds * (np.sqrt(np.sum(ps**2) / dim) - np.sqrt(dim)))
            
            # Update covariance
            y_norm = (mean - old_mean) / sigma
            
            # Rank-one update
            C += c1 * (np.outer(y_norm, y_norm) + (1 - hsig_val * cs * (2 - cs)) * pc @ pc.T - C)
            
            # Rank-mu update
            if cmu > 0:
                y_mu = np.array([(y_sorted[i] - old_mean) / sigma for i in range(mu)])
                # Weights for rank-mu
                w_mu = weights[:mu]
                C += cmu * (np.sum(w_mu[:, None, None] * np.outer(y_mu, y_mu), axis=0) - C)
                
            # Ensure positive definiteness and update B, D
            # For dim=5, we can do a full eigen decomposition or Cholesky
            # C should be symmetric
            C = (C + C.T) / 2.0
            
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
                B, D = L[:, :].T, np.diag(L).copy()
                # Actually, C = L L^T. So B = L^T, D = diag(L)
                # y = B @ (D * z) = L^T @ (diag(L) * z)
                # This matches the standard form.
            except np.linalg.LinAlgError:
                # If not positive definite, add identity
                C = C + 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                    B, D = L[:, :].T, np.diag(L).copy()
                except:
                    # Fallback to identity
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
        
        return best_x, best_f
