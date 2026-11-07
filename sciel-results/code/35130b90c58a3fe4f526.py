"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for 5D Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # 300s safety margin out of 330s
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Initial mean: center of the box is a good guess for shifted functions
        m = np.zeros(d)
        # Initial step size: cover a large portion of the box initially
        sigma = 2.0
        
        # Population size: lambda = 4 + floor(3 log d) is standard, for d=5 -> 4+3*1.6=8.8 -> 12? 
        # Standard is often 4 + 3*log(d). For d=5, lambda ~ 12.
        # Let's use lambda = 12 for d=5.
        lambda_pop = 4 + int(3 * np.log(d))
        
        # Weights for selection
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # State variables
        pc = np.zeros(d)
        ps = np.zeros(d)
        cs = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1)) * c_sigma
        # Initial covariance matrix is identity
        C = np.eye(d)
        # Inverse of C
        C_inv = np.eye(d)
        # Cholesky factor of C
        B = np.eye(d)
        D = np.ones(d)
        
        # Precompute constants
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Initial evaluation
        best_x = m.copy()
        best_f = func(m)
        eval_count = 1
        
        # Main loop
        while eval_count < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            ys = np.zeros((lambda_pop, d))
            for i in range(lambda_pop):
                # Sample from N(0, C)
                z = np.random.standard_normal(d)
                y = m + sigma * (B @ (D * z))
                ys[i] = y
            
            # Evaluate population
            fs = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if time.time() > deadline:
                    # Fill remaining with best known to maintain array shape if needed, 
                    # but we break out of loop logic shortly
                    fs[i] = best_f
                    continue
                fs[i] = func(ys[i])
                eval_count += 1
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = ys[i].copy()
            
            # Sort by fitness
            sort_idx = np.argsort(fs)
            ys = ys[sort_idx]
            fs = fs[sort_idx]
            
            # Update mean
            y_w = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)
            m = m + sigma * y_w
            
            # Update evolution path for covariance matrix adaptation
            # Transform y_w to C^-1/2 space
            # We need C^-1/2. We have B and D such that C = B diag(D^2) B^T.
            # So C^-1/2 = B diag(1/D) B^T
            # Actually, the standard update uses the "normalized" step
            # z_w = C^-1/2 (y_w)
            # Since y_w is already in the "step" space (scaled by sigma in the update),
            # let's stick to the standard CMA-ES update rules.
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update ps
            # Need C^-1/2 y_w
            # C = B diag(D^2) B^T
            # C^-1/2 = B diag(1/D) B^T
            # z_w = B @ ( (1/D) * (B.T @ y_w) )
            z_w = B @ ((1.0 / D) * (B.T @ y_w))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w
            
            # Update covariance matrix C
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (eval_count - 1)))) < (1.4 + 2 / (d + 1)) * chi_d
            if not np.isfinite(hsig): hsig = True # safety
            
            delta_hs = (1 - hsig) * c_c * (2 - c_c)
            
            rank_one = np.outer(ps, ps)
            rank_mu = np.sum(weights[:, np.newaxis] * (ys[:mu] * ys[:mu].T), axis=0)
            
            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is positive definite (numerical stability)
            # Symmetrize
            C = (C + C.T) / 2.0
            try:
                # Recompute Cholesky
                L = np.linalg.cholesky(C)
                # Update B and D
                # C = B diag(D^2) B^T
                # This requires eigen decomposition or updating from Cholesky
                # For simplicity and speed in low dim, we can just recompute eigen each step
                # or use the update from L.
                # Let's use eigen decomposition for robustness in this small dim
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip negative eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                # If C is not PD, reset to identity or last valid state
                C = np.eye(d)
                B = np.eye(d)
                D = np.ones(d)
            
            # Update step size sigma
            # sigma = sigma * exp( c_sigma / chi_d * ( ||ps||/chi_d - 1 ) )
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(ps) / chi_d - 1))
            
            # Clamp sigma to avoid exploding or vanishing
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update best_x if mean is better? No, we track best from evaluations.
            
            # Convergence check
            if np.max(np.abs(y_w)) < 1e-8:
                break
                
        return best_x, best_f

# EVOLVE-BLOCK-END
