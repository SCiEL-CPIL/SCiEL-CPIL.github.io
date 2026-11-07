import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (mu + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initialization
        rng = np.random.default_rng(42)
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        calls = 0
        best_f = np.inf
        best_x = None
        gen = 0
        
        start_time = time.time()
        max_time = 290.0
        
        while calls < budget:
            if time.time() - start_time > max_time:
                break
                
            gen += 1
            
            # Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reinitialize C
                C = np.eye(n)
                L = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                sigma = 2.0
                continue

            # Generate population
            Y = rng.standard_normal((lambda_, n))
            X = x_mean[None, :] + sigma * (Y @ L.T)
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lambda_
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            # Sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            Y_sorted = Y[order]
            
            # Weighted mean in Y-space and X-space
            Y_new = np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            x_new = np.sum(w[:, None] * X[:mu], axis=0)
            
            # Update p_sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * Y_new
            
            # h_sigma condition
            # Approximation of the norm of ps in the history
            # Using the standard threshold
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * gen * c_sigma))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
            
            # Update p_c
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * Y_new
            
            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            # Rank-mu update: sum w_i Y_i Y_i^T
            # Y_sorted[:mu] is (mu, n). w is (mu,).
            # We need sum w_i (Y_i Y_i^T)
            Y_mu = Y_sorted[:mu]
            rank_mu = np.sum(w[:, None, None] * Y_mu[:, :, None] * Y_mu[:, None, :], axis=0)
            
            C = (1 - c_1 - c_mu + delta_h) * C + c_1 * np.outer(pc, pc) + c_mu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Update mean
            x_mean = x_new

            if calls >= budget:
                break
                
        return best_x, best_f
