"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin for 300s limit + overhead

        d = self.dim
        # CMA-ES parameters
        # lambda = 4 + floor(3 * log(d))
        lam = 4 + int(3 * np.log(d))
        
        # Initial sigma
        # Heuristic for f11 (Discus): elongated landscape. 
        # Start relatively small to avoid overshooting the narrow valley, 
        # but large enough to find the basin.
        # Standard heuristic: sigma = 0.3 * (b - a) / d or similar.
        # Box is [-5, 5], width 10. 
        # Let's try 0.5 as a reasonable start for 5D.
        sigma = 0.5
        
        # Initial mean: center of the box is a safe bet, 
        # but minimizer is in [-4, 4]. 
        # Random start might be better to avoid symmetry traps if any, 
        # but center is usually good for symmetric-ish functions.
        # However, f11 has a specific structure. 
        # Let's start at 0.
        x_mean = np.zeros(d)
        
        # Covariance matrix
        C = np.eye(d)
        
        # Step size adaptation state
        ps = np.zeros(d)
        pc = np.zeros(d)
        
        # Weights for selection
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Chi norm for step size control
        chi_n = d**0.5 * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Damping for pc
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        
        # Initial evaluation
        x0 = np.clip(x_mean, -5, 5)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls_made = 1
        
        # Pre-compute sqrt of C for sampling? 
        # We need to update C and then take sqrt. 
        # For small d (5), this is fast.
        
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            # We need sqrt(C). 
            # Efficient way: eigendecomposition or Cholesky.
            # Since C is symmetric positive definite, Cholesky is good.
            # But C is updated by rank-mu update, which preserves symmetry.
            # Let's use np.linalg.cholesky. 
            # If it fails (numerical issues), add small epsilon to diagonal.
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-12 * np.eye(d)
                L = np.linalg.cholesky(C)
            
            # Sample lambda points
            # z ~ N(0, I)
            Z = np.random.randn(lam, d)
            # y = sigma * L @ z^T ? No, x = mean + sigma * L @ z
            # Actually, standard CMA-ES: x_i = mean + sigma * (L @ z_i)
            # Wait, usually it's x_i = mean + sigma * B * d * z_i where C = B D^2 B^T.
            # Using Cholesky L such that C = L L^T, then x = mean + sigma * L @ z.
            
            X = x_mean[None, :] + sigma * (L @ Z.T).T
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lam
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Compute C update terms
            # y_inv = L^{-1} (x_new - x_mean) / sigma
            # Solve L y_inv = (x_new - x_mean) / sigma
            diff = (x_new - x_mean) / sigma
            # Solve L v = diff
            try:
                y_inv = np.linalg.solve(L, diff)
            except np.linalg.LinAlgError:
                y_inv = np.linalg.lstsq(L, diff, rcond=None)[0]
                
            # Update ps
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mueff / d)
            ps = (1 - c_sigma) * ps + c_sigma_sqrt * y_inv
            
            # Check step size adaptation
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lam))) < (1.4 + 2 / (d + 1)) * chi_n:
                sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update pc
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lam)))) < (1.4 + 2 / (d + 1)) * chi_n
            h_sigma = 1.0 if h_sigma else 0.0
            
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_inv
            
            # Update C
            # Rank 1 update
            C = (1 + c_1 * h_sigma * (mueff - 1) - c_1 - c_mu) * C \
                + c_1 * (pc[:, None] * pc[None, :] + (1 - h_sigma) * c_c * (2 - c_c) * np.eye(d)) \
                + c_mu * np.sum(weights[:mu][:, None, None] * (X_sorted[:mu] - x_mean)[:, :, None] * (X_sorted[:mu] - x_mean)[:, None, :], axis=0) / sigma**2
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Add small regularization if needed
            # Check eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-12:
                    C += (1e-12 - eigvals.min()) * np.eye(d)
            except:
                pass
            
            x_mean = x_new
            
        return best_x, best_f

# EVOLVE-BLOCK-END
