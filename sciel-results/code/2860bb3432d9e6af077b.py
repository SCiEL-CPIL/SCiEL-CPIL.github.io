"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for Black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Weights
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
        c_1 = (2 - mu_eff/dim + c_c) / ((dim + 1.3)**2 + mu_eff + c_c)
        c_mu = np.min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        inv_sigma = 1.0
        
        # Precompute square root of C
        def compute_sqrt(C):
            # Use Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                # Fallback to eigenvalue decomposition if not positive definite
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 0)
                return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

        B, D = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 0))
        B = B @ np.diag(D)
        
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        calls_made = 1
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate
            fys = [func(y) for y in ys]
            calls_made += lambda_
            
            # Sort by fitness
            idx = np.argsort(fys)
            ys = [ys[i] for i in idx]
            fys = [fys[i] for i in idx]
            
            # Update best
            if fys[0] < best_fx:
                best_fx = fys[0]
                best_x = ys[0].copy()
            
            # Weighted mean
            x_new = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            
            # Update covariance adaptation
            y_diff = x_new - x
            x = x_new
            
            # Update evolution paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * y_diff
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_diff
            
            # Heaviside function
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(dim+1)) * chi_n else 0
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            delta_d = (h_sigma - 1 + c_c) * c_c
            outer = np.outer(y_diff, y_diff)
            C = (1 + c_c * (c_c - 2) + c_1 * delta_d) * C + c_1 * outer
            for i in range(mu):
                diff = (ys[i] - x) / sigma
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Re-orthogonalize and update B, D periodically to prevent numerical drift
            if calls_made % 10 == 0:
                # Symmetrize
                C = (C + C.T) / 2
                # Ensure positive definite
                try:
                    B, D = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 0))
                    B = B @ np.diag(D)
                except:
                    pass
                    
        return best_x, best_fx
# EVOLVE-BLOCK-END
