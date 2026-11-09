import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        c_sigma = (4 + 3 * n / lambda_) / ((n + 2)**2 + 3 * lambda_)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_1 = 2 / ((n + 1.3)**2 + lambda_)
        c_mu = 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + lambda_)
        c_cm = (4 + lambda_ / n) / ((n + 4)**2 + lambda_)
        
        chi_N = np.sqrt(n) * (1 - 1 / (4 * n**2) + 1 / (21 * n**4))
        
        x0 = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x0
        best_f = np.inf
        calls = 0
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            z = np.random.standard_normal((lambda_, n))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
            
            y = z @ L.T
            x_offspring = x0 + sigma * y
            f_offspring = np.array([func(x) for x in x_offspring])
            calls += lambda_
            
            idx_best = np.argmin(f_offspring)
            if f_offspring[idx_best] < best_f:
                best_f = f_offspring[idx_best]
                best_x = x_offspring[idx_best].copy()
            
            order = np.argsort(f_offspring)
            x_sorted = x_offspring[order]
            f_sorted = f_offspring[order]
            
            x0_old = x0
            x0 = np.sum(w[:, None] * x_sorted[:mu], axis=0)
            
            y_sigma = (x_sorted[:mu] - x0_old) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * L.T @ y_sigma
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1 if norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // lambda_ + 1))) < (1.4 + 2 / (n + 1)) * chi_N else 0
            if h_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_N - 1))
            
            pc = (1 - c_cm) * pc + h_sigma * np.sqrt(c_cm * (2 - c_cm) * mu_eff) * (x0 - x0_old) / sigma
            
            # Covariance update
            # Sum of outer products of selected y's
            # y_i = (x_i - x0_old) / sigma
            y_selected = (x_sorted[:mu] - x0_old) / sigma
            sum_w_yy = np.einsum('i,ij,ik->jk', w, y_selected, y_selected)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * c_cm * (2 - c_cm) * C) + c_mu * sum_w_yy
            
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
        
        return best_x, best_f
