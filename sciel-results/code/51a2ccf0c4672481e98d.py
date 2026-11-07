"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # Parameters
        # Standard CMA-ES settings for 5D
        lambda_pop = 4 * int(np.floor(3 + np.log(dim)))  # 4 * 4 = 16
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Initial state
        # Start near center, but with some spread
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        def update_C_sqrt(C, C_sqrt):
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                # Fallback to eigen decomposition if not positive definite
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-12)
                return eigenvectors @ np.diag(np.sqrt(eigenvalues))

        def update_C_sqrt_inv(C, C_sqrt_inv):
            # Inverse square root via eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))

        # Initial evaluations
        # We need to evaluate the initial population to start the loop
        # But CMA-ES usually generates offspring from the mean.
        # Let's do a simple random initialization for the first step or just start the loop.
        # To save budget, we'll let the loop handle the first generation.
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Pre-allocate storage for current generation
        ys = np.zeros((lambda_pop, dim))
        fs = np.zeros(lambda_pop)
        
        # We need to be careful with the budget.
        # Total budget is 500.
        # Each generation costs lambda_pop evaluations.
        # 500 / 16 = 31 generations.
        
        for gen in range(budget // lambda_pop + 1):
            if time.time() - start_time > 290:
                break
            
            # Generate offspring
            for i in range(lambda_pop):
                if time.time() - start_time > 290:
                    break
                z = np.random.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                # Clip to box
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                
                # Track best
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = y.copy()
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update mean
            x_mean_old = x_mean
            x_mean = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)
            
            # Update step size control variables
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_sqrt_inv @ (x_mean - x_mean_old)
            step_vec = (x_mean - x_mean_old) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ step_vec)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_eff) * step_vec
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step_vec
            
            # h_sigma = (||ps|| / sqrt(1 - (1-c_sigma)^{2(gen+1)})) / chi_n < 1.4 + 2/(dim+1)
            # Approximation for h_sigma
            h_sigma = (np.linalg.norm(ps) / chi_n) < (1.4 + 2.0 / (dim + 1))
            
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - h_sigma) * c_c * (2 - c_c) * C) 
            #    + c_mu * sum(weights_i * (y_i - x_mean_old)(y_i - x_mean_old).T)
            
            diff = (x_mean - x_mean_old) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            for i in range(mu):
                diff_i = (ys[i] - x_mean_old) / sigma
                C += c_mu * weights[i] * np.outer(diff_i, diff_i)
            
            # Ensure positive definiteness and update square roots
            C = 0.5 * (C + C.T)
            C_sqrt = update_C_sqrt(C, C_sqrt)
            C_sqrt_inv = update_C_sqrt_inv(C, C_sqrt_inv)
            
            # If we run out of budget mid-generation, the loop will stop anyway
            if gen * lambda_pop + lambda_pop >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
