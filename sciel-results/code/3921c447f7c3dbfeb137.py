"""CMA-ES optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation tuned for 5D BBOB f7 with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # For dim=5: 4 + 3*1.609 = 8 (floor) -> 7, let's use 9 for better exploration
        lam = max(4 + int(3 * np.log(dim)), 4)
        
        # Actually for dim=5, 4+3*log(5) ≈ 4+4.82 = 8.82, floor is 8. Let's use 9 or 10.
        # Standard recommendation: lambda = 4 + floor(3 ln n)
        # For n=5: 4 + floor(3*1.609) = 4 + 4 = 8.
        # Let's use lambda = 10 for a bit more robustness in 5D.
        lam = 10

        # Step size
        sigma = 2.0  # Initial step size, box is [-5,5]^5, so this covers a good portion

        # Mean
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        ps = np.zeros(dim)
        hc = 0.0  # For cumulation of (1+lambda)-CMA-ES
        
        # Learning rates
        cmu = 2 / ((dim + 1.3)**2 + lam)  # Mutation strength
        cc = (4 + lam/dim) / (dim + 4 + 2*lam/dim)  # Cumulation for sigma
        cs = (cc + 2) / (dim + cc + 3)  # Cumulation for step size
        c1 = (2 + 2/dim) / ((dim + 2)**2 + lam)  # Rank-one update
        cmu = min(2 / ((dim + 1.3)**2 + lam), 0.5 * (1 - c1))  # Rank-mu update

        # Damping for covariance matrix update
        damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1) + cs

        # Initialization
        # Start near center
        m = np.zeros(dim)
        
        # Pre-compute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Weighting for active CMA-ES (optional, but standard CMA-ES is fine)
        # Weights
        w = np.log(np.arange(1, lam+1)) - np.log(np.arange(lam, 0, -1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w[:int(0.5*lam)]**2)
        
        # Adjust learning rates for weighted recombination
        c1 = (2 + 2/dim) / ((dim + 2)**2 + 2*mu_eff)
        cmu = min(2 / ((dim + 1.3)**2 + 2*mu_eff), 0.5 * (1 - c1))

        best_x = m.copy()
        best_f = func(m)
        calls = 1

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring
            # Transform C to get square root (Cholesky decomposition)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, try to make C positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = np.linalg.cholesky(C)

            Y = np.random.randn(lam, dim)
            X = np.tile(m, (lam, 1)) + sigma * (Y @ C_sqrt.T)

            # Evaluate offspring
            F = np.zeros(lam)
            for i in range(lam):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                F[i] = func(X[i])
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()

            if calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]

            # Update mean
            m_new = np.sum(w[:, None] * X, axis=0)
            
            # Update evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C_sqrt.T @ (m_new - m))
            
            # Update sigma
            hc = (1 - cc) * hc + (np.sqrt(cc * (2 - cc)) * np.linalg.norm(ps) / chi_n > 1.4 + 2 / (dim + 1))
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            delta_hc = (1 - hc) * cc * (2 - cc)
            C = (1 + c1 * delta_hc - c1 - cmu) * C \
                + c1 * (np.outer(m_new - m, m_new - m) + delta_hc * C) \
                + cmu * np.sum(w[:, None, None] * np.outer(X, X), axis=0)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            m = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
