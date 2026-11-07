import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.RandomState(42)
        
        # CMA-ES Parameters
        lam = int(4 + 3 * np.log(d)) if d > 1 else 4
        mu = lam // 2
        
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        
        sigma = 1.0
        mean = np.zeros(d)
        C = np.eye(d)
        C_inv_sqrt = np.eye(d)
        
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        evals = 0
        best_x = mean.copy()
        best_f = np.inf
        
        while evals < budget:
            n_eval_this_gen = min(lam, budget - evals)
            if n_eval_this_gen <= 0:
                break
            
            X = np.zeros((n_eval_this_gen, d))
            for i in range(n_eval_this_gen):
                z = rng.standard_normal(d)
                x = mean + sigma * (C_inv_sqrt @ z)
                x = np.clip(x, -5, 5)
                X[i] = x
            
            F = np.zeros(n_eval_this_gen)
            for i in range(n_eval_this_gen):
                F[i] = func(X[i])
                evals += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
                
                if evals >= budget:
                    break
            
            if evals >= budget:
                break
            
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            X_selected = X_sorted[:mu]
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * X_selected, axis=0)
            
            step = (mean - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ step)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals / lam))) < (1.4 + 2 / (d + 1)) * chi_d else 0
            
            c1_term = c_1 * (h_sigma - 1 + c_c * (2 - c_c)) * np.outer(p_c, p_c)
            
            diffs = X_selected - old_mean[np.newaxis, :]
            rank_mu_term = c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * diffs[:, :, np.newaxis] * diffs[:, np.newaxis, :], axis=0)
            
            C = (1 + c_1 * (h_sigma - 1) + c_mu) * C + c1_term - rank_mu_term
            C = (C + C.T) / 2
            
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L.T)
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(d)
                try:
                    L = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(L.T)
                except np.linalg.LinAlgError:
                    C = np.eye(d)
                    C_inv_sqrt = np.eye(d)
                    sigma = 1.0
                    p_sigma = np.zeros(d)
                    p_c = np.zeros(d)

        return best_x, best_f
