import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280

        rng = np.random.default_rng(42)
        call_count = [0]
        best_f = np.inf
        best_x = np.zeros(dim)

        def safe_call(x):
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            return func(x)

        # CMA-ES Parameters
        N = 4 + int(3 * np.log(dim)) # Population size
        N = max(N, 8) # Ensure minimum population
        lambda_ = N // 2 # Number of best points used for update
        mu = np.zeros(dim)
        sigma = 2.0 # Initial step size
        C = np.eye(dim)
        # Pre-compute weights
        weights = np.log(lambda_ + 0.5) - np.log(np.arange(1, lambda_ + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff / 2)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Trackers
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))

        for gen in range(budget // N + 1):
            if time.time() > deadline or call_count[0] >= budget:
                break
            
            # Sample
            Y = rng.multivariate_normal(mu, sigma**2 * C, size=N)
            X = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([safe_call(x) for x in X])
            
            # Update Best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                
            # Sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update Mean
            m_new = np.sum(weights[:, np.newaxis] * X[:lambda_], axis=0)
            
            # Update Sigma
            y_sigma = (np.sqrt(np.sum(C)) / sigma) * (m_new - mu)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update Covariance
            y_c = (m_new - mu) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            delta_heav = 1 if np.linalg.norm(p_c)**2 / (1 - (1 - c_c)**(2 * (gen + 1))) > (1 + 4 / dim) * chi_n**2 else 0
            
            C = (1 + c_1 * delta_heav - c_1 - c_mu) * C + \
                c_1 * (p_c @ p_c.T + delta_heav * chi_n**2 * C) + \
                c_mu / sigma**2 * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(X[:lambda_] - mu, X[:lambda_] - mu), axis=0)
            
            mu = m_new
            
            # Check for convergence
            if sigma < 1e-10:
                break

        return best_x, best_f
