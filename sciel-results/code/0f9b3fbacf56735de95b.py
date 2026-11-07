import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        rng = np.random.default_rng(0)
        
        # CMA-ES Parameters
        # lambda = 4 + int(3 * np.log(dim)) is standard, but for small N, sometimes larger is better?
        # Standard is usually 4 + 3 ln N. For N=5, 4 + 4 = 8.
        # Let's stick to standard.
        lambda_ = 4 + int(3 * np.log(dim))
        if lambda_ > budget:
            lambda_ = budget # Safety, though 8 < 500
        
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial state
        # Start at 0
        m = np.zeros(dim)
        C = np.eye(dim)
        sigma = 2.0 # Initial step size
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Pre-compute Cholesky of C
        C_sqrt = np.linalg.cholesky(C)
        inv_C_sqrt = np.linalg.inv(C_sqrt)
        
        x_best = m.copy()
        fx_best = func(x_best)
        calls = 1
        
        # Loop
        while calls < budget and time.time() < deadline:
            # Check if we have enough budget for a full generation
            if budget - calls < lambda_:
                # If not enough for a full generation, we could do a partial one or stop.
                # Standard CMA-ES assumes full generations.
                # If remaining budget is small, we might just break to avoid partial update complexity, 
                # or handle it. Given 500/8 = 62.5, we will likely stop after 62 generations (496 calls).
                # The remaining 4 calls are wasted if we break. 
                # However, implementing partial generations is complex. 
                # Let's just break if we can't do a full generation to keep it simple and robust.
                break
            
            # Sample population
            population = np.zeros((lambda_, dim))
            for i in range(lambda_):
                # z ~ N(0, I)
                z = rng.standard_normal(dim)
                # y = sigma * C_sqrt @ z
                y = sigma * (C_sqrt @ z)
                x_i = m + y
                population[i] = x_i
            
            # Evaluate
            f_values = np.zeros(lambda_)
            for i in range(lambda_):
                f_values[i] = func(population[i])
                calls += 1
            
            # Sort by fitness
            sorted_indices = np.argsort(f_values)
            population_sorted = population[sorted_indices]
            f_values_sorted = f_values[sorted_indices]
            
            # Update best
            if f_values_sorted[0] < fx_best:
                fx_best = f_values_sorted[0]
                x_best = population_sorted[0].copy()
            
            # Selection
            x_w = population_sorted[:mu]
            
            # Mean update
            m_old = m.copy()
            m = m + c_mu * (weights @ (x_w - m))
            
            # Path updates
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt @ inv_C_sqrt @ (m - m_old) / sigma)
            
            # p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt @ inv_C_sqrt @ (m - m_old) / sigma)
            
            # Covariance update
            # Rank one
            y_sigma = (m - m_old) / sigma
            C = (1 + c_1 * (np.dot(p_sigma, p_sigma) / c_sigma - 1) + c_mu * (np.dot(p_c, p_c) / c_c - 1)) * C \
                + c_1 * np.outer(y_sigma, y_sigma)
            
            # Rank mu
            diff = x_w - m_old
            diff_scaled = diff / sigma
            C += c_mu * (weights @ (diff_scaled[:, np.newaxis] * diff_scaled))
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Step size adaptation
            # chi_n = sqrt(N) * (1 - 1/(4N) + 1/(21N^2))
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Check if p_sigma norm is too large
            # E ||p_sigma|| = sqrt(1 - (1-c_sigma)^2) * chi_n approx chi_n
            # If ||p_sigma|| / chi_n > 1.4 + 2/(dim+1), we increase sigma? No, we adjust sigma.
            # Standard CSA:
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            # d_sigma is usually 1 or tuned. Let's use 1 for simplicity or standard value.
            # Standard d_sigma = 1.
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma * (norm_p_sigma / chi_n - 1))
            
            # Clip sigma
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
                inv_C_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                # If Cholesky fails, C is not positive definite.
                # Add small identity to make it PD
                C = C + 1e-6 * np.eye(dim)
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                    inv_C_sqrt = np.linalg.inv(C_sqrt)
                except np.linalg.LinAlgError:
                    # Reset C
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    inv_C_sqrt = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    sigma = 1.0
                    continue

        return x_best, fx_best
