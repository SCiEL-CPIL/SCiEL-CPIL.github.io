"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style Weierstrass function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Standard population size: 4 + 3*log(n)
        lam = int(4 + 3 * np.log(dim))
        if lam < 4:
            lam = 4
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_m = (2 + np.sqrt(mu_eff / dim)) / (dim + 2 + mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(0.4 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial state
        m = np.zeros(dim)
        C = np.eye(dim)
        sigma = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute Cholesky factor
        L = np.linalg.cholesky(C)
        
        total_calls = 0
        best_f = np.inf
        best_x = m.copy()
        
        while total_calls < budget:
            # Generate population
            # Check if we can afford a full generation
            if lam > budget - total_calls:
                # If not, reduce population size or break? 
                # Better to just break if we can't do a full generation to maintain CMA-ES stability,
                # or do a partial one. Standard CMA-ES requires full population for update.
                # Given 500 budget and lam=9, we have 55 full generations.
                # 55*9 = 495. 5 calls left.
                # Let's just break if remaining budget < lam.
                break
            
            # Sample
            Y = np.random.randn(lam, dim)
            # X = m + sigma * Y @ L.T (since L is lower triangular, Y @ L.T is standard)
            # Actually, if C = L L.T, then z = L T y is such that z.T z = y.T L L.T y = y.T C y.
            # So X = m + sigma * (Y @ L.T)
            X = m + sigma * (Y @ L.T)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            total_calls += lam
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            # Sort
            sort_idx = np.argsort(F)
            X_sorted = X[sort_idx]
            F_sorted = F[sort_idx]
            
            # Mean update
            # y = (X_sorted - m) / sigma
            y = (X_sorted[:mu] - m) / sigma
            y_mean = np.sum(weights[:, None] * y, axis=0)
            
            # Update path length estimator ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.inv(L) @ y_mean
            
            # Update covariance
            # Check if ps is too large
            h_sigma = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (total_calls // lam + 1))) < 
                          (1 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)) + 1e-10) else 0.0
            
            # Update path cm
            pc = (1 - c_c) * pc + c_c * h_sigma * y_mean
            
            # Update mean
            m = m + c_m * pc
            
            # Update covariance
            # Rank-one update
            delta_sigma = c_1 * (h_sigma * (np.outer(pc, pc) - C) + (1 - h_sigma) * c_c * (1 - c_c) * C)
            
            # Rank-mu update
            # y_sorted = (X_sorted[:mu] - m) / sigma
            # Note: We should use the OLD m for the update? 
            # Standard CMA-ES updates C using the points relative to the OLD mean m.
            # So y_sorted is correct as computed above.
            
            # Weights for rank-mu
            # w_mu = weights[:mu] * (1 - c_1)
            w_mu = weights * (1 - c_1)
            
            # Sum of outer products
            # C += c_mu * sum(w_mu_i * outer(y_i, y_i))
            # y_sorted is shape (mu, dim)
            # w_mu is shape (mu,)
            # We need sum(w_mu_i * y_i[:, None] * y_i[None, :])
            
            # Efficient calculation:
            # (Y * w_mu[:, None]).T @ Y
            C += c_mu * (y.T @ (y * w_mu[:, None]))
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
            # Update sigma (CSA)
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Update Cholesky factor
            # We need to update L such that L L.T = C
            # Use np.linalg.cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, add small regularization
                C += 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)
                
            # Safety check: if sigma becomes too small or large, stop
            if sigma < 1e-10 or sigma > 1e10:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
