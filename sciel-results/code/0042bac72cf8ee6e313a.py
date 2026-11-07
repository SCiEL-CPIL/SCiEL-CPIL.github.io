"""Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # mu = number of parents
        # lambda = population size
        # Standard choices for dim=5
        lambda_pop = 2 * n + 5  # ~15 for n=5
        mu = lambda_pop // 2     # ~7
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_cov = (2 + 2 * mu_eff / (n + 2)) / (n + 1 + 3 * mu_eff)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial State
        x_mean = np.zeros(n)
        sigma = 2.0  # Initial step size
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        calls = 0
        best_x = None
        best_f = float('inf')

        # Main Loop
        while calls + lambda_pop <= budget:
            # 1. Sample Population
            # Compute C^{1/2} using eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            z = rng.standard_normal((lambda_pop, n))
            y = z @ C_sqrt.T  # y = C^{1/2} z
            pop = x_mean + sigma * y
            
            # Evaluate
            f_pop = np.array([func(x) for x in pop])
            calls += lambda_pop
            
            # Update Best
            idx_min = np.argmin(f_pop)
            if f_pop[idx_min] < best_f:
                best_f = f_pop[idx_min]
                best_x = pop[idx_min].copy()
            
            # 2. Sort by fitness
            idx_sorted = np.argsort(f_pop)
            y_sorted = y[idx_sorted]
            f_sorted = f_pop[idx_sorted]
            
            # 3. Weighted Mean of Selected y's
            y_w = np.sum(w[:, None] * y_sorted[:mu], axis=0)
            
            # 4. Update Mean
            x_mean = x_mean + c_c * sigma * y_w
            
            # 5. Update Path p_sigma
            # h_sigma condition
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 2 / (n + 1)) * c_sigma * chi_n else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            
            # 6. Update Path p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # 7. Update Covariance C
            # C = (1 - c_cov) C + c_cov * (sum_{i=1}^mu w_i y_i y_i^T + (1 - h_sigma) * c_c * (2 - c_c) * p_c p_c^T)
            # Note: The standard update includes the rank-one term of y_w as well.
            # C = (1 - c_cov) C + c_cov * (y_w y_w^T + (1 - h_sigma) * c_c * (2 - c_c) * p_c p_c^T)
            # However, the rank-mu update is more standard for the main term.
            # Let's use: C = (1 - c_cov) C + c_cov * (C_rank_mu + (1 - h_sigma) * c_c * (2 - c_c) * p_c p_c^T)
            # Where C_rank_mu = sum(w_i y_i y_i^T)
            
            C_rank_mu = np.sum((w[:, None, None] * y_sorted[:mu][:, None, :] * y_sorted[:mu][:, None, :].T), axis=0)
            C_p_c = (1 - h_sigma) * c_c * (2 - c_c) * np.outer(p_c, p_c)
            
            C = (1 - c_cov) * C + c_cov * (C_rank_mu + C_p_c)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            
            # 8. Update Sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

        # Local Refinement with remaining budget
        if best_x is not None and calls < budget:
            x = best_x.copy()
            f_x = best_f
            step = 0.5
            for _ in range(budget - calls):
                dx = rng.standard_normal(n) * step
                x_new = np.clip(x + dx, -5, 5)
                f_new = func(x_new)
                calls += 1
                if f_new < f_x:
                    x = x_new
                    f_x = f_new
                    best_x = x.copy()
                    best_f = f_x
                else:
                    step *= 0.8
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
