import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290

        best_x = None
        best_f = float('inf')
        eval_count = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Random Initialization
        n_init = min(int(0.1 * budget), 50)
        rng = np.random.default_rng(42)
        
        for _ in range(n_init):
            if eval_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            f = func(x)
            eval_count += 1
            update_best(x, f)

        if eval_count >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES
        lam = 10
        weights = np.log(lam/2 - np.arange(lam) + 1)
        weights /= weights.sum()
        mu = int(lam / 2)
        mu_eff = 1.0 / np.sum(weights[:mu]**2)
        
        c_c = (3 + mu_eff/n) / (n + 8 + mu_eff/n)
        c_sigma = (1 + 2 * max(0, np.sqrt((mu_eff/n - 1)/(n + 1))) - c_c) / (n + 1.5)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff))
        
        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        def update_C_sqrt_inv(C):
            w, V = np.linalg.eigh(C)
            w = np.clip(w, 1e-12, None)
            return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

        while eval_count + lam <= budget and time.time() < deadline:
            ys = []
            fs = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = x_mean + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                eval_count += 1
                ys.append(y)
                fs.append(fy)
                update_best(y, fy)
                
            if len(ys) < lam:
                break
                
            idx = np.argsort(fs)
            ys_sorted = [ys[i] for i in idx]
            
            x_new = np.sum(np.array(ys_sorted[:mu]) * weights[:mu, None], axis=0)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_new - x_mean) / (sigma * chi_n)
            
            # Update p_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (eval_count // lam)))) < (1.4 + 2 / (n + 1)) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Update C
            delta_h = c_1 * (np.outer(p_c, p_c) - C)
            delta_mu = np.zeros((n, n))
            for i in range(mu):
                diff = ys_sorted[i] - x_mean
                delta_mu += weights[i] * np.outer(diff, diff)
            delta_mu *= c_mu
            
            C = (1 + c_1 * (2 * h_sigma - 1) + c_mu - c_1) * C + delta_h + delta_mu
            
            # Update sigma
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            x_mean = x_new
            
            # Update C_sqrt_inv
            C_sqrt_inv = update_C_sqrt_inv(C)
            
            if sigma < 1e-10:
                break

        return best_x, best_f
