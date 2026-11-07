"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for BBOB f6 (Attractive Sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # CMA-ES parameters
        # lambda = 9 for n=5 (standard recommendation is 4 + 3*log(n) ~ 10, but 9 is common)
        # Let's use lambda = 4 + int(3 * np.log(n))
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

        # Initial state
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        hs = (1.4 + 2.0 / (n + 1)) * np.sqrt((mu_eff - 1) + 1)
        
        # Track best
        best_x = m.copy()
        best_f = func(best_x)
        calls = 1

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Check if we have enough budget for a full generation
            if budget - calls < lam:
                break

            # Sampling
            Z = np.random.standard_normal((lam, n))
            # Transformation: x = m + sigma * B * D * z
            # B is orthogonal, D is diagonal
            X = m + sigma * (Z @ (B * D).T)
            
            # Clip to box to ensure validity? The task says func clips, but it's safer to clip ourselves to avoid issues?
            # "func(x) clips a finite length-5 vector to the box" -> so we don't need to clip, but it doesn't hurt.
            # However, if we clip, the distribution becomes distorted. Better to let func handle it.
            
            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                F[i] = func(X[i])
                calls += 1
            
            # Sort
            indices = np.argsort(F)
            F = F[indices]
            X = X[indices]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Recombination
            m_old = m
            m = np.sum([w[i] * X[i] for i in range(mu)], axis=0)
            
            # Step size adaptation (CSA)
            # y = (m - m_old) / sigma
            y = (m - m_old) / sigma
            # z = D^-1 B^-1 y
            z = (B.T @ y) / D
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            
            # Update sigma
            # hs is the threshold
            # chs = 1 + 2*max(0, (mu_eff - 1)/(n + 1)) - 1
            chs = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) - 1
            if np.linalg.norm(p_sigma) / hs < (1 + 2 * max(0, (mu_eff - 1) / (n + 1))) * (1 - c_sigma):
                sigma *= np.exp(c_sigma / chs * (np.linalg.norm(p_sigma) / hs - 1))
                
            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * y
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c + (1 - (1 - c_c)^2) * I) + c_mu * sum(w_i * y_i * y_i^T)
            # Where y_i are the top mu steps
            
            # Compute sum of outer products of top mu steps
            # y_i = (X[i] - m_old) / sigma
            sum_outer = np.zeros((n, n))
            for i in range(mu):
                y_i = (X[i] - m_old) / sigma
                sum_outer += w[i] * np.outer(y_i, y_i)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * np.eye(n)) + c_mu * sum_outer
            
            # Re-orthonormalize C (eigen-decomposition)
            # C should be symmetric, but numerical errors might creep in
            C = (C + C.T) / 2
            eigenvalues, B = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            D = np.sqrt(eigenvalues)
            # B is already orthogonal from eigh
            # Note: eigh returns eigenvalues in ascending order.
            # We need to ensure B and D are consistent.
            # x = m + sigma * B * D * z
            # This requires B to be the matrix of eigenvectors and D the sqrt of eigenvalues.
            # The order must match.
            
            # Convergence check?
            # If sigma is too small, stop.
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
