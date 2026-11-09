import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (n + 2) - 1))
        
        rng = np.random.default_rng()
        
        # Initial point
        x = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        gen = 0
        while calls + lam <= budget:
            # 1. Sample population
            Z = rng.standard_normal((lam, n))
            # y = x + sigma * C @ z
            # Z is (lam, n). We want (lam, n) result.
            # C @ z for each z.
            # (Z @ C.T) gives (lam, n) where row i is Z[i] @ C.T = (C @ Z[i].T).T
            # Since C is symmetric, C @ Z[i].T is the vector.
            Y = x[None, :] + sigma * (Z @ C)
            
            F = np.array([func(y) for y in Y])
            calls += lam
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = Y[min_idx].copy()
            
            # 2. Sort
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # 3. Update mean
            y_w = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            x = x + y_w
            
            # 4. Update covariance matrix
            # Compute delta sigma
            # Time constant for p_sigma is 1/c_sigma
            # The norm of p_sigma is compared to chi_n * sqrt(1 - (1-c_sigma)^(2*gen))
            # Actually, the standard h_sigma check uses the expected norm.
            # Let's use the standard formula:
            # h_sigma = ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*gen)) < (1.4 + 2/(n+1)) * chi_n
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ y_w) / sigma
            
            # Check h_sigma
            # gen is the number of generations completed so far?
            # We are at the end of generation gen.
            # The time constant factor is (1-c_sigma)^(2 * (calls/lam)) roughly?
            # Let's use the generation count.
            gen += 1
            # The expected norm of p_sigma after gen steps is chi_n * sqrt(1 - (1-c_sigma)^(2*gen))
            # Wait, p_sigma is a random walk.
            # The condition is:
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen))) < (1.4 + 2 / (n + 1)) * chi_n
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w / sigma
            
            # Rank-one and rank-mu update
            # C = (1 + c_1 * delta_sigma + c_mu) * C
            # delta_sigma = (||p_sigma|| - chi_n) / (chi_n * (1 + c_sigma))
            delta_sigma = (np.linalg.norm(p_sigma) - chi_n) / (chi_n * (1 + c_sigma))
            
            C = (1 + c_1 * delta_sigma + c_mu) * C \
                + c_1 * (p_c @ p_c - (1 - c_c) * C) \
                + c_mu * np.sum(weights[:, None] * (Y_sorted[:mu] - x) @ (Y_sorted[:mu] - x).T, axis=0) / sigma**2
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # 5. Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clip sigma
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # If sigma is extremely small, we can break early?
            # No, we want to use the budget.
            
        return best_x, best_f
