"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        N = 4 + int(3 * np.log(d))
        # Step size
        sigma = 2.0
        # Initial mean
        m = rng.uniform(-5, 5, d)
        # Initial covariance
        C = np.eye(d)
        # Eigen-decomposition cache
        D_inv = np.ones(d)
        B = np.eye(d)
        
        # Adaptation rates
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        chi_N = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d**2))
        
        # Evaluation tracking
        count = 0
        best_x = m.copy()
        best_f = func(best_x)
        count += 1
        
        start_time = time.time()
        deadline = start_time + 280
        
        # Main loop
        while count < budget:
            if time.time() > deadline:
                break
                
            # Update eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D_inv = 1.0 / np.sqrt(eigvals)
            B = eigvecs
            
            # Sample offspring
            Y = B @ (D_inv * rng.standard_normal((N, d)))
            X = np.array([m + sigma * y for y in Y])
            
            # Evaluate
            F = np.array([func(np.clip(x, -5, 5)) for x in X])
            count += N
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            Y = Y[idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Update mean
            m_new = m + sigma * np.sum(weights[:, None] * Y[:mu], axis=0)
            
            # Update step size (CSA)
            z_sigma = (B @ (D_inv * (m_new - m))) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (count - budget))) < (1.4 + 2 / (d + 1)) * chi_N)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            
            # Update covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma
            C = (1 + c_1 * (c_mu + 2) - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            C += c_mu * np.sum(weights[:mu][:, None, None] * np.outer(Y[:mu], Y[:mu]), axis=0)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            try:
                # Regularize if needed
                eigvals_check = np.linalg.eigvalsh(C)
                if np.any(eigvals_check < 1e-12):
                    C += 1e-12 * np.eye(d)
            except:
                C = np.eye(d)
                
            m = m_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
