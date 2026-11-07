"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES Optimizer for BBOB-style functions.
    
    Uses the Covariance Matrix Adaptation Evolution Strategy.
    This method adapts the sampling distribution (mean, covariance, step size)
    to the local landscape geometry, which is highly effective for rotated,
    non-separable functions like Schaffers F7.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # Safety margin vs 330 s wall-clock kill

        rng = np.random.default_rng(42)

        # ---- Global Best Bookkeeping -------------------------------------
        x_best = None
        f_best = np.inf
        n_calls = 0

        def record(x, f):
            nonlocal x_best, f_best, n_calls
            if f < f_best:
                f_best = f
                x_best = x.copy()
            n_calls += 1
            return f_best

        # ---- CMA-ES Parameters -------------------------------------------
        # Population size: lambda = 4 + floor(3 * ln(n)) is standard, but for small n,
        # we can use a slightly larger population to ensure diversity early on.
        # For n=5, standard lambda = 4 + 3*ln(5) ~ 8.8 -> 8.
        # Let's use lambda = 10 for robustness in low dimensions.
        lambda_ = 10
        
        # Initial step size: 2.0 is a common heuristic for [-5, 5] box.
        # It allows covering a significant portion of the space initially.
        sigma = 2.0
        
        # Initial mean: Start at the center of the box.
        x_mean = np.zeros(dim)
        
        # Initial covariance: Identity matrix.
        C = np.eye(dim)
        
        # Adaptation parameters (standard CMA-ES values)
        mu = lambda_ // 2  # Number of weights used for update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff)))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation of the mean
        f_mean = func(x_mean)
        record(x_mean, f_mean)
        
        if n_calls >= budget or time.time() > deadline:
            return x_best, f_best

        # Pre-compute Cholesky decomposition of C (initially I)
        # We will update C and recompute Cholesky each generation.
        
        while n_calls < budget and time.time() < deadline:
            # 1. Sample lambda_ individuals
            # v = C^{1/2} * z
            # Use Cholesky for stability
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_sigma = np.zeros(dim)
                L = np.eye(dim)
            
            Z = rng.standard_normal((lambda_, dim))
            # Transform to x
            Y = np.dot(Z, L.T) * sigma
            X = x_mean[None, :] + Y
            
            # Clip to box [-5, 5]
            X = np.clip(X, -5.0, 5.0)
            
            # 2. Evaluate all lambda_ individuals
            F = np.empty(lambda_)
            for i in range(lambda_):
                F[i] = func(X[i])
                record(X[i], F[i])
                if n_calls >= budget or time.time() > deadline:
                    break
            
            if n_calls >= budget or time.time() > deadline:
                break

            # If we broke out early due to budget, we might not have all evaluations.
            # CMA-ES requires a full population for stable updates.
            # If we don't have a full population, we stop the loop.
            if n_calls >= budget or time.time() > deadline:
                break
            
            # 3. Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # 4. Update mean
            # x_mean_new = sum(weights_i * x_sorted_i)
            x_mean_new = np.dot(weights, X_sorted[:mu, :])
            
            # 5. Update evolution paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^{-1/2} * (x_mean_new - x_mean) / sigma
            # We need C^{-1/2}. We can use the inverse of the Cholesky factor L.
            # C^{-1/2} = (L^{-1})^T
            try:
                Linv = np.linalg.inv(L)
                C_inv_sqrt = Linv.T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
            
            diff_mean = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff_mean)
            
            # h_sigma indicator
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim * dim))
            h_sigma = 1 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (n_calls + 1) / lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff_mean
            
            # 6. Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(weights_i * (y_i y_i^T))
            # where y_i = (x_sorted_i - x_mean) / sigma
            
            Y_sorted = (X_sorted[:mu, :] - x_mean[None, :]) / sigma
            
            # Weighted outer products
            # Use einsum for efficiency: c_mu * sum_i w_i * y_i y_i^T
            # Y_sorted shape (mu, dim), weights shape (mu,)
            # outer = np.einsum('i,ij,ik->jk', weights, Y_sorted, Y_sorted)
            
            # To avoid numerical issues, we can also add a small regularization if needed,
            # but standard CMA-ES is usually stable.
            
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.einsum('i,ij,ik->jk', weights, Y_sorted, Y_sorted)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            
            # Symmetrize C to prevent drift
            C = (C + C.T) / 2.0
            
            # 7. Update step size sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # 8. Update mean
            x_mean = x_mean_new

        return x_best, f_best
# EVOLVE-BLOCK-END
