"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lambda_pop = 4 * int(1 + 3 * dim)  # Population size
        mu = lambda_pop // 2                 # Number of best points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_cov = (2 + 2 * mu_eff / dim) / (dim**1.8 + mu_eff + 13)
        
        # Initial values
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial evaluation of mean (optional but good for baseline)
        # Actually, CMA-ES usually starts by generating the first population.
        # To be efficient, we can start with a random point to have a valid best_x immediately if needed.
        # But standard CMA-ES generates population from mean.
        # Let's just run the loop.
        
        # We need to ensure we don't exceed budget.
        # Each generation uses lambda_pop evaluations.
        # Total generations approx budget / lambda_pop.
        
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            Y = np.array([rng.multivariate_normal(np.zeros(dim), C, 1).flatten() for _ in range(lambda_pop)])
            # Actually, using rng.multivariate_normal with covariance C is slow and can be unstable.
            # Better to use Cholesky decomposition or eigendecomposition.
            # Let's use eigendecomposition for stability and speed, updated only when C changes.
            # However, updating C requires inverse square root.
            
            # Let's stick to a simpler, robust implementation using numpy's linalg.
            # For dim=5, it's fast enough.
            
            # Generate standard normal samples
            Z = rng.standard_normal((lambda_pop, dim))
            # Transform to C distribution
            # x_i = x_mean + sigma * (C_sqrt @ Z_i)
            # We maintain C_sqrt_inv? No, we need C_sqrt.
            # Let's maintain C and compute C_sqrt via eigendecomp each step? 
            # Or maintain C_sqrt_inv and update it?
            # Standard CMA-ES updates C and then computes C^{-1/2} for the next step.
            
            # Let's use a slightly simplified but effective CMA-ES implementation.
            # Since dim is small (5), we can afford to compute eigendecomposition each step if needed, 
            # but it's better to update C and then use a stable method to sample.
            
            # Let's re-implement the loop core with proper CMA-ES logic.
            
            # 1. Sample
            # We need C^{1/2}. Let's compute it from C.
            # Eigenvalues and eigenvectors of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-10)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            Y = Z @ C_sqrt.T  # Each row is a sample from N(0, C)
            X = x_mean + sigma * Y
            X = np.clip(X, -5, 5)
            
            # 2. Evaluate
            F = np.array([evaluate(x) for x in X])
            
            # Check time
            if time.time() > deadline:
                break
                
            # 3. Update
            # Sort indices by fitness
            idx = np.argsort(F)
            X_best = X[idx[:mu]]
            F_best = F[idx[:mu]]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * X_best, axis=0)
            
            # Update sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^{-1/2} * (x_mean_new - x_mean) / sigma
            # Need C^{-1/2}
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            y_sigma = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_sigma)
            
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (budget - calls)))) \
                      / (1 + 1e-16) < (1 + 2 / (dim + 1)) * chi_n
            
            # Update C
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Rank-mu update
            v = (X_best - x_mean) / sigma
            delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.linalg.norm(p_c)**2 / dim
            rank_one = np.outer(p_c, p_c)
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(v[i], v[i])
                
            C = (1 + c_cov * (delta_h - 1)) * C + c_cov * (rank_mu - rank_one)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            x_mean = x_mean_new
            
            # Optional: Reset if sigma becomes too small or large
            if sigma < 1e-15:
                break
            if sigma > 1e10:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
