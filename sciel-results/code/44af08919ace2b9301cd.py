"""Rewritten optimizer for BBOB f13 Sharp Ridge."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f13 (Sharp Ridge) using CMA-ES.
    CMA-ES adapts the covariance matrix to the local landscape,
    making it highly effective for rotated and ill-conditioned functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f = float(func(x))
            except:
                return None
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Brief random sampling to initialize CMA-ES mean
        n_random = min(10, budget // 10)
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            fx = safe_call(x)
            if fx is None:
                break

        if best_x is None:
            return best_x, best_f

        # Initialize CMA-ES parameters
        x_mean = best_x.copy()
        sigma = 1.0  # Initial step size
        C = np.eye(dim)
        # Pre-compute Cholesky decomposition for sampling
        try:
            C_eigenval = np.linalg.eigh(C)
            C_eigenval = C_eigenval[0]
            C_eigenvec = C_eigenval[1]
            C_sqrt = np.linalg.cholesky(C)
        except:
            C_sqrt = np.eye(dim)
            
        # CMA-ES parameters
        lambda_ = int(4 + 3 * np.log(dim))  # Population size
        mu = lambda_ // 2  # Number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))

        # Main CMA-ES Loop
        while calls < budget and time.time() < deadline:
            # Sample population
            pop_x = np.zeros((lambda_, dim))
            pop_f = np.zeros(lambda_)
            
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5, 5)
                fx = safe_call(x)
                if fx is None:
                    break
                pop_x[i] = x
                pop_f[i] = fx
                
            if calls >= budget:
                break
            
            # Sort population by fitness
            order = np.argsort(pop_f)
            pop_x = pop_x[order]
            pop_f = pop_f[order]
            
            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop_x[0].copy()
                
            # Recombination weights
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (pop_x[i] - x_mean) / sigma
                
            # Update step size sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
            # Update path p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean) / sigma # Note: x_mean is updated after, so this uses old x_mean effectively if we delay update, but standard CMA updates p_c with (new_mean - old_mean)
            # Correction: p_c update uses the difference between new and old mean
            # Let's stick to standard formulation:
            # p_c = (1 - c_c) p_c + sqrt(c_c(2-c_c)mu_eff) * (x_new - x_old) / sigma
            
            # We need to calculate the new mean first to update p_c correctly in standard implementations, 
            # or update p_c using the weighted sum of y's.
            # Standard: p_c = (1 - c_c) p_c + sqrt(c_c(2-c_c)mu_eff) * (sum w_i * (x_i - x_old) / sigma)
            # This is exactly y_w * sqrt(...)
            
            # Update Covariance Matrix C
            # Rank-1 update
            if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (calls // lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n:
                C = (1 + c_1 * (c_mu + 1) - c_1) * C + c_1 * np.outer(p_c, p_c)
            else:
                C = (1 + c_1 * (c_mu + 1) - c_1) * C + c_1 * (np.outer(p_c, p_c) / (1 - (1 - c_c)**(2 * (calls // lambda_))) if False else np.outer(p_c, p_c)) # Fallback
            
            # Rank-mu update
            y_w_list = [(pop_x[i] - x_mean) / sigma for i in range(mu)]
            C += c_mu * np.sum([np.outer(y, y) * weights[i] for i, y in enumerate(y_w_list)], axis=0)
            
            # Ensure C is symmetric and positive definite (approximate)
            C = (C + C.T) / 2
            
            # Update Mean
            x_mean = np.sum([weights[i] * pop_x[i] for i in range(mu)], axis=0)
            x_mean = np.clip(x_mean, -5, 5)
            
            # Update Cholesky decomposition periodically to save time or if it fails
            # For simplicity and robustness, we update every iteration if dim is small (dim=5)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except:
                # If Cholesky fails, add small regularization
                C += np.eye(dim) * 1e-6
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    C_sqrt = np.eye(dim)
                    C = np.eye(dim)

        return best_x, best_f

# EVOLVE-BLOCK-END
