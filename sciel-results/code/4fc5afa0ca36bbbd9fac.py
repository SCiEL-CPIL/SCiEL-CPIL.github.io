import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Initial point
        x0 = rng.uniform(-4, 4, n)
        x_best = x0.copy()
        f_best = func(x0)
        self.calls += 1
        
        if self.calls >= budget:
            return x_best, f_best

        # CMA-ES parameters
        sigma = 2.0
        C = np.eye(n)
        mu = x_best.copy()
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Population size
        lam = 4 + int(3 * np.log(n))
        lam = max(lam, 8)
        
        # Precompute weights and constants
        def update_params(lam):
            weights = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
            weights = weights / np.sum(weights)
            mu_eff = 1.0 / np.sum(weights[:min(lam // 2, lam)]**2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            return weights, mu_eff, c_sigma, c_c, c_1, c_mu, chi_n

        weights, mu_eff, c_sigma, c_c, c_1, c_mu, chi_n = update_params(lam)
        
        # Stagnation counter
        gen_no_improve = 0
        threshold = 10
        
        while self.calls < budget:
            # Check if we can afford a full population
            if self.calls + lam > budget:
                # Try to use remaining budget if it's at least 2
                if budget - self.calls >= 2:
                    lam = budget - self.calls
                    weights, mu_eff, c_sigma, c_c, c_1, c_mu, chi_n = update_params(lam)
                else:
                    break

            # Sample
            x_off = np.zeros((lam, n))
            for i in range(lam):
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                z = rng.standard_normal(n)
                x_off[i] = mu + sigma * (L @ z)
            
            # Clip to bounds
            x_off = np.clip(x_off, -5, 5)
            
            # Evaluate
            f_off = np.array([func(x) for x in x_off])
            self.calls += lam
            
            # Sort
            idx = np.argsort(f_off)
            x_off = x_off[idx]
            f_off = f_off[idx]
            
            # Update best
            if f_off[0] < f_best:
                f_best = f_off[0]
                x_best = x_off[0].copy()
                gen_no_improve = 0
            else:
                gen_no_improve += 1
            
            # Check for stagnation
            if gen_no_improve >= threshold:
                # Increase population size
                new_lam = int(lam * 1.5)
                if new_lam < lam:
                    new_lam = lam + 2
                lam = new_lam
                
                # Reset distribution
                mu = x_best.copy()
                sigma = 1.0
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                
                # Recalculate parameters
                weights, mu_eff, c_sigma, c_c, c_1, c_mu, chi_n = update_params(lam)
                
                gen_no_improve = 0
                continue # Skip update, start new generation with new params
            
            # Standard CMA-ES Update
            
            # 1. Update mean
            k = min(lam // 2, lam)
            mu_new = np.sum(weights[:k] * x_off[:k], axis=0)
            
            # 2. Update step size (p_sigma)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C^(-1/2) * (mu_new - mu) / sigma)
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C)).T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (mu_new - mu) / sigma)
            
            # 3. Update correlation (p_c)
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (mu_new - mu) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * ((mu_new - mu) / sigma)
            
            # 4. Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * p_c p_c^T + c_mu * sum(wi * yi yi^T)
            # where yi = (xi - mu) / sigma
            
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (self.calls - self.budget + 1)))) \
                      / chi_n < (1.4 + 2 / (n + 1)) * (1 - c_sigma)
            
            # Note: The h_sigma calculation above is approximate because we don't track generation count precisely for the power term.
            # A simpler heuristic for h_sigma is often used:
            # h_sigma = 1.0 if (np.linalg.norm(p_sigma) / chi_n) < (1.4 + 2 / (n + 1)) * (1 - c_sigma) else 0.0
            
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / chi_n) < (1.4 + 2 / (n + 1)) * (1 - c_sigma) else 0.0

            C = (1 - c_1 - c_mu + h_sigma * c_1 * (1 - c_c)) * C \
                + c_1 * (p_c + h_sigma * c_c * (mu - mu_new) / sigma) @ (p_c + h_sigma * c_c * (mu - mu_new) / sigma).T \
                + c_mu * np.sum(weights[:k, None] * ((x_off[:k] - mu) / sigma) @ ((x_off[:k] - mu) / sigma).T, axis=0)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # 5. Update sigma
            # sigma = sigma * exp(c_sigma / chi_n * (||p_sigma|| / chi_n - 1))
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # 6. Update mean
            mu = mu_new

        return x_best, f_best
