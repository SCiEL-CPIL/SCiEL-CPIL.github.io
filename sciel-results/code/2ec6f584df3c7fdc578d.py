import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        calls = 0
        dim = self.dim
        budget = self.budget

        def call(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            calls += 1
            return func(x)

        n = dim
        lam = 4 + int(3 * np.log(n))
        if lam < 10:
            lam = 10
            
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        f_mean = evaluate(x_mean)
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()

        while calls_made < budget and time.time() < deadline:
            # Sample
            z = np.random.normal(size=(lam, n))
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            x_pop = x_mean + sigma * (B @ (D * z.T).T)
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate
            f_pop = np.array([evaluate(x) for x in x_pop])
            
            if calls_made >= budget:
                break
                
            idx = np.argsort(f_pop)
            x_pop = x_pop[idx]
            f_pop = f_pop[idx]
            
            if f_pop[0] < best_f:
                best_f = f_pop[0]
                best_x = x_pop[0].copy()
                
            x_new = np.sum(weights[:, np.newaxis] * x_pop[:mu], axis=0)
            
            # Update sigma
            B_inv, D_inv = np.linalg.eigh(C)
            D_inv = 1.0 / np.sqrt(np.maximum(D_inv, 1e-12))
            C_sqrt_inv = B_inv @ np.diag(D_inv) @ B_inv.T
            y_new = C_sqrt_inv @ (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_new
            norm_p_sigma = np.linalg.norm(p_sigma)
            if norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1):
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(n) - 1))
                
            # Update C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            C = (1 + c_1 * (c_mu + 1) - c_1) * C + c_1 * np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = C_sqrt_inv @ (x_pop[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            C += c_mu * rank_mu
            C = (C + C.T) / 2
            x_mean = x_new
            
            if sigma < 1e-10:
                break

        # Polish
        if calls_made < budget and time.time() < deadline:
            step = 0.1
            for _ in range(10):
                improved = False
                for d in range(dim):
                    for direction in (1, -1):
                        x_new = best_x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        if calls_made >= budget:
                            break
                        f_new = func(x_new)
                        calls_made += 1
                        if f_new < best_f:
                            best_x = x_new
                            best_f = f_new
                            improved = True
                        else:
                            break
                step *= 0.7
                if not improved:
                    break

        return best_x, best_f
