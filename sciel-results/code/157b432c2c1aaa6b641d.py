"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with time guard and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0
        rng = np.random.default_rng(0)

        global_best_x = None
        global_best_fx = np.inf
        total_calls = 0

        # CMA-ES parameters
        lambda_pop = 6
        mu = lambda_pop // 2
        weights = np.zeros(mu)
        for i in range(1, mu + 1):
            weights[i - 1] = np.log(mu + 0.5) - np.log(i)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / (dim ** 1.8 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / (dim ** 2 + mu_eff))
        d_sigma = 1.0 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma

        def update_global(y, fy):
            nonlocal global_best_x, global_best_fx
            if fy < global_best_fx:
                global_best_fx = fy
                global_best_x = y.copy()

        def run_cma_es(x_mean, sigma, max_calls):
            nonlocal total_calls
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            while total_calls < max_calls and time.time() < deadline:
                # Check if we can afford a full generation
                if budget - total_calls < lambda_pop:
                    break
                
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: reset C to identity if it becomes singular
                    C = np.eye(dim)
                    L = np.eye(dim)
                    # Maybe reduce sigma?
                    sigma *= 0.5

                z = rng.standard_normal((lambda_pop, dim))
                y = x_mean[None, :] + sigma * (z @ L.T)
                y = np.clip(y, -5, 5)
                
                fy = np.array([func(yi) for yi in y])
                total_calls += lambda_pop
                
                idx = np.argsort(fy)
                y_sorted = y[idx]
                fy_sorted = fy[idx]
                
                update_global(y_sorted[0], fy_sorted[0])
                
                y_parents = y_sorted[:mu]
                x_new_mean = np.sum(weights[:, None] * y_parents, axis=0)
                
                diff = (x_new_mean - x_mean) / sigma
                try:
                    v = np.linalg.solve(L, diff)
                except np.linalg.LinAlgError:
                    v = diff # Fallback

                h_sigma = 1.0
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
                
                Y_diff = y_parents - x_mean
                C_sum = Y_diff.T @ (Y_diff * weights[:, None])
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * C_sum
                C = (C + C.T) / 2.0
                C += 1e-12 * np.eye(dim)
                
                p_sigma_norm = np.linalg.norm(p_sigma)
                sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                x_mean = x_new_mean

            # If we exited due to insufficient budget for full generation, 
            # we can use (1+1)-ES for the remaining calls
            while total_calls < max_calls and time.time() < deadline:
                y = np.clip(x_mean + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                update_global(y, fy)
                if fy < global_best_fx:
                    x_mean = y
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        # Restart 1
        x_mean = rng.uniform(-5, 5, dim)
        sigma = 2.0
        run_cma_es(x_mean, sigma, budget // 2)

        # Restart 2
        if total_calls < budget:
            x_mean = rng.uniform(-5, 5, dim)
            sigma = 2.0
            run_cma_es(x_mean, sigma, budget)

        return global_best_x, global_best_fx

# EVOLVE-BLOCK-END
