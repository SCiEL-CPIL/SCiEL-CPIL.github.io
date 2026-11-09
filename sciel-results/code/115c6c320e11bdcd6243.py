"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the BBOB-style function f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        
        # Parameters for CMA-ES
        dim = self.dim
        budget = self.budget
        
        # Initial population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Weights: logarithmic weights
        w = np.log(np.arange(1, lambda_ + 1))
        w /= w.sum()
        mu = int(np.floor(lambda_ / 2))
        w = w[:mu]
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2)**2 + mu_eff))
        
        # Initial sigma
        sigma = 1.0
        
        # Initial mean: uniform random in [-5, 5]^5
        x_mean = rng.uniform(-5, 5, dim)
        
        # Covariance matrix
        C = np.eye(dim)
        # Square root of C
        C_sqrt = np.eye(dim)
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        h_sigma = 1.0
        
        # Track best solution
        best_x = x_mean.copy()
        best_f = func(x_mean)
        
        # Initial evaluations
        calls_made = 1
        
        while calls_made < budget and time.time() < deadline:
            # Generate population
            y = np.array([rng.multivariate_normal(np.zeros(dim), C) for _ in range(lambda_)])
            x = x_mean + sigma * y
            # Clip to box [-5, 5]
            x = np.clip(x, -5, 5)
            
            # Evaluate
            f = np.array([func(xi) for xi in x])
            calls_made += lambda_
            
            # Sort by fitness
            idx = np.argsort(f)
            x = x[idx]
            f = f[idx]
            
            # Update best
            if f[0] < best_f:
                best_f = f[0]
                best_x = x[0].copy()
            
            # Weighted recombination
            x_new = np.sum(w[:, None] * x[:mu], axis=0)
            
            # Update covariance matrix (rank-one and rank-mu update)
            y_old = C_sqrt @ (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_old
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Heuristic to avoid numerical problems
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lambda_ + 1) / lambda_)) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)):
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # Update C
            C = (1 + c_1 * h_sigma * (p_c @ p_c) - c_1) * C + c_mu * np.sum(w[:, None, None] * np.outer(x[:mu] - x_mean, x[:mu] - x_mean) / (sigma**2), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lambda_ + 1) / lambda_)) - np.sqrt((dim + 1) / (dim - 1)) * (1 - h_sigma)))
            
            # Update mean
            x_mean = x_new
            
            # Update C_sqrt (using Cholesky or SVD, SVD is more robust for small steps)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            except np.linalg.LinAlgError:
                # Fallback if SVD fails
                C_sqrt = np.eye(dim)
                C = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
