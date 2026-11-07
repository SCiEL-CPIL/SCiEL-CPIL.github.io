"""Surrogate-based optimizer for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        if lam < 10:
            lam = 10
            
        mu = lam // 2
        
        # Weights
        weights = np.zeros(mu)
        w_sum = 0
        for i in range(mu):
            weights[i] = np.log(mu + 1) - np.log(i + 1)
            w_sum += weights[i]
        weights /= w_sum
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # Initial state
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = mean.copy()
        best_f = np.inf
        calls_made = 0
        generation = 0
        
        # Initial evaluation
        f_val = func(mean)
        calls_made += 1
        if f_val < best_f:
            best_f = f_val
            best_x = mean.copy()

        while calls_made < budget:
            if time.time() > deadline:
                break
            
            remaining = budget - calls_made
            current_lam = min(lam, remaining)
            
            if current_lam <= 0:
                break

            # Sample
            Y = rng.standard_normal((current_lam, dim))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim) * 1e-6
                L = np.linalg.cholesky(C)
            
            X = mean[None, :] + sigma * (Y @ L.T)
            
            # Evaluate
            F = np.empty(current_lam)
            for i in range(current_lam):
                F[i] = func(X[i])
                calls_made += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
                
                if time.time() > deadline:
                    break
            
            if calls_made >= budget:
                break

            # Update
            order = np.argsort(F)
            X_sorted = X[order]
            
            # New mean
            mean_new = np.zeros(dim)
            for i in range(mu):
                mean_new += weights[i] * X_sorted[i]
            
            # Step size control
            y_old = (mean_new - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_old
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (generation + 1))) / np.sqrt(dim)
            
            # Covariance control
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_old
            
            # Covariance update
            C = (1 + c_1 * (h_sigma * (2 - h_sigma) - 1) - c_1) * C
            C += c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            C += c_mu * sum(weights[i] * np.outer(X_sorted[i] - mean, X_sorted[i] - mean) for i in range(mu))
            
            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(dim) - 1))
            
            mean = mean_new
            C = (C + C.T) / 2
            
            # Regularization
            if generation % 5 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            generation += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
