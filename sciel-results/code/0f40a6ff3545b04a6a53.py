import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0
        rng = np.random.default_rng(42)
        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x))
                calls[0] += 1
                return val
            except:
                calls[0] += 1
                return float('inf')

        best_x = np.zeros(dim)
        best_f = float('inf')

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        sigma = 2.0 
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        nu_sigma = 1 - c_sigma

        while calls[0] < budget and time.time() < deadline:
            x_pop = []
            f_pop = []
            
            for i in range(lam):
                if calls[0] >= budget:
                    break
                z = rng.standard_normal(dim)
                # Compute C_inv_sqrt efficiently via SVD every step? No, just use Cholesky or SVD once per step.
                # For dim=5, SVD is fast.
                U, s, Vt = np.linalg.svd(C)
                C_inv_sqrt = (U / np.sqrt(s)) @ Vt
                x = x_mean + sigma * (C_inv_sqrt @ z)
                x = np.clip(x, -5.0, 5.0)
                f = safe_call(x)
                x_pop.append(x)
                f_pop.append(f)

            if len(f_pop) < lam:
                break

            indices = np.argsort(f_pop)
            x_pop = np.array(x_pop)[indices]
            f_pop = np.array(f_pop)[indices]
            
            if f_pop[0] < best_f:
                best_f = f_pop[0]
                best_x = x_pop[0].copy()

            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * x_pop[:mu], axis=0)
            
            y_sigma = (x_mean - x_mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            y_c = (x_mean - x_mean_old) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            rank_c = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(x_pop[:mu] - x_mean_old, x_pop[:mu] - x_mean_old) / sigma**2, axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_c + (1 - nu_sigma * (np.linalg.norm(p_sigma)**2 / chi_n**2 - 1) * rank_c) + c_mu * rank_mu)
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            # Re-orthogonalize
            U, s, Vt = np.linalg.svd(C)
            C = (U * s) @ Vt

            if sigma < 1e-12:
                break

        return best_x, best_f
