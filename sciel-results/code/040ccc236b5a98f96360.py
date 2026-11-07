import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.deadline = 320.0  # 10s safety margin before 330s kill

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = max(4, 4 + int(3 * np.log(n)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 - mu_eff / 2) / (n + 8)
        c_mu = min(2 * (2 - mu_eff / (n + 2)) / ((n + 2) * (n + 2) + mu_eff), 2 / (n + 3))
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(c_mu, 2 * (1 - c_1 - c_c) / (n + 2))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = m.copy()
        best_f = func(best_x)
        calls = 1
        gen = 0
        
        while calls < budget:
            if time.time() > self.deadline:
                break
            
            gen += 1
            
            # Generate population
            pop = np.zeros((lam, n))
            for i in range(lam):
                z = np.random.randn(n)
                y = B @ (D * z)
                pop[i] = m + sigma * y
                pop[i] = np.clip(pop[i], -5.0, 5.0)
            
            # Evaluate population
            fvals = np.zeros(lam)
            for i in range(lam):
                if time.time() > self.deadline:
                    break
                fvals[i] = func(pop[i])
                calls += 1
                if calls >= budget:
                    break
            
            if time.time() > self.deadline or calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            fvals_sorted = fvals[idx]
            pop_sorted = pop[idx]
            
            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()
                
            # Update mean
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * (pop_sorted[i] - m) / sigma
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            
            # h_sigma condition
            # t is gen
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * gen / (n + c_sigma)))
            num = np.linalg.norm(p_sigma)
            thresh = (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) * (1 - (1 - c_sigma)**(2 * gen / (n + c_sigma))))
            h_sigma = 1.0 if (num / denom < thresh) else 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update mean
            m = m + c_sigma * sigma * y_w
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance matrix
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            for i in range(mu):
                y_i = (pop_sorted[i] - m) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update B and D
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                B = np.eye(n)
                D = np.ones(n)
                C = np.eye(n)

        return best_x, best_f
