"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES Optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n)) 
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / (w ** 2).sum()
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Initialization
        rng = np.random.default_rng()
        m = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        p_c = np.zeros(n)
        p_s = np.zeros(n)
        D = np.linalg.cholesky(C)
        
        best_x = m.copy()
        best_f = func(m)
        evals = 1
        
        try:
            while evals < budget:
                if time.time() > deadline:
                    break
                
                # Sample lambda points
                z = rng.standard_normal((lam, n))
                y = m[:, None] + sigma * (z @ D.T)
                y = np.clip(y, -5, 5)
                
                # Evaluate
                fvals = np.array([func(yi) for yi in y])
                evals += lam
                
                # Sort by fitness
                idx = np.argsort(fvals)
                y = y[idx]
                fvals = fvals[idx]
                
                # Update Best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = y[0].copy()
                
                # Update Mean
                m_new = w @ y[:mu]
                
                # Update p_sigma
                p_s = (1 - c_sigma) * p_s + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (D.T @ (m_new - m) / sigma)
                
                # Update p_c
                # Standard h_sig check
                norm_p_s = np.linalg.norm(p_s)
                # Approximate threshold
                h_sig = 1 if (norm_p_s / chi_n) < (1.4 + 2 / (n + 1)) else 0
                p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - m) / sigma
                
                # Update Covariance
                delta_h = (1 - h_sig) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
                for i in range(mu):
                    y_i = (y[i] - m) / sigma
                    C += c_mu * w[i] * np.outer(y_i, y_i)
                
                # Update Sigma
                sigma *= np.exp(c_sigma / chi_n * (norm_p_s / chi_n - 1))
                
                m = m_new
                # Ensure positive definiteness
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) <= 0:
                    C = C + 1e-10 * np.eye(n)
                D = np.linalg.cholesky(C)
                
        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
