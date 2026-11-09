"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Default population size
        lambda_ = max(4 + int(3 * np.log(n)), 4)
        
        # Initial mean
        m = np.zeros(n)
        
        # Initial step size
        sigma = 2.0
        
        # Initial covariance matrix
        C = np.eye(n)
        
        # Eigenvalues of C (for adaptation)
        eigval, eigvec = np.linalg.eigh(C)
        
        # Step size control parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = m.copy()
        best_f = float('inf')
        
        evals_used = 0
        
        # Pre-compute some constants
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))
        
        # Main loop
        while evals_used < budget:
            # Generate population
            ys = np.zeros((lambda_, n))
            xs = np.zeros((lambda_, n))
            fs = np.zeros(lambda_)
            
            for i in range(lambda_):
                if evals_used >= budget:
                    break
                    
                # Sample from N(0, C)
                # Use Cholesky or eigen decomposition for sampling
                # For simplicity and speed in small D, we can use the current eigvec/eigval
                # z ~ N(0, I)
                z = np.random.standard_normal(n)
                y = eigvec @ (np.sqrt(eigval) * z)
                
                x = m + sigma * y
                x = np.clip(x, -5.0, 5.0)
                
                xs[i] = x
                ys[i] = y
                fs[i] = func(x)
                evals_used += 1
                
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = x.copy()
            
            if evals_used >= budget:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            xs = xs[order]
            ys = ys[order]
            fs = fs[order]
            
            # Update mean
            m_new = np.sum(weights[:, np.newaxis] * xs[:mu], axis=0)
            
            # Update covariance matrix
            # Calculate successful steps
            y_w = (xs[:mu] - m) / sigma
            C_new = C
            for i in range(mu):
                C_new += weights[i] * np.outer(y_w[i], y_w[i])
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (eigvec @ (np.sqrt(eigval) ** -1 @ (y_w.T @ weights)))
            
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals_used // lambda_ + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (eigvec @ (np.sqrt(eigval) ** -1 @ (y_w.T @ weights)))
            
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            
            C = (1 + c_1 + c_mu - c_c * (2 - c_c) + delta_hsig) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            C += c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * y_w[:, :, np.newaxis] * y_w[:, np.newaxis, :], axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update eigen decomposition
            eigval, eigvec = np.linalg.eigh(C)
            # Ensure positive definite
            eigval = np.maximum(eigval, 1e-10)
            
            # Update step size
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            m = m_new
            
            # Update best if new mean is better (optional, but we track global best)
            # The loop above already updates best_x/best_f when sampling.
            
            # Convergence check
            if np.max(fs) - np.min(fs) < 1e-8:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
