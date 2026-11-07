"""CMA-ES based optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Robust CMA-ES for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # Population size: lambda = 4 + 3 * ln(n)
        lambda_ = 4 + int(3 * np.log(dim))
        if lambda_ < 2:
            lambda_ = 2
        
        # Recombination
        mu = lambda_ // 2
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / w.sum()

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialization
        # Start at center of the box [-5, 5]^d
        mean = np.zeros(dim)
        sigma = 2.0
        
        # Covariance matrix C = B @ diag(D^2) @ B.T
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute C^{-1/2}
        C_inv_sqrt = np.eye(dim)

        best_x = mean.copy()
        best_fx = func(best_x)
        calls = 1

        # Main Loop
        while calls + lambda_ <= budget:
            # Generate population
            Z = np.random.standard_normal((lambda_, dim))
            # x_i = mean + sigma * B @ (D * z_i)
            # Vectorized: Y = (B * D) @ Z.T  -> shape (dim, lambda_)
            Y = (B * D) @ Z.T
            X = mean + sigma * Y.T
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lambda_

            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_fx:
                best_fx = F[min_idx]
                best_x = X[min_idx].copy()

            # Sort
            idx = np.argsort(F)
            X_sorted = X[idx]

            # Update mean
            mean_old = mean.copy()
            mean = np.sum(w[:, None] * X_sorted[:mu], axis=0)

            # Update paths
            diff = (mean - mean_old) / sigma
            
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff)
            
            # h_sigma (using simplified threshold)
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1 + 2 / (dim + 1)) * chi_n else 0.0

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)

            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff)

            # Update Covariance
            # Rank-1 update
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Rank-mu update
            Y_selected = (X_sorted[:mu] - mean_old) / sigma
            C += c_mu * (Y_selected.T * w) @ Y_selected

            # Symmetry
            C = (C + C.T) / 2

            # Eigendecomposition
            try:
                D_sq, B = np.linalg.eigh(C)
                D_sq = np.maximum(D_sq, 1e-12)
                D = np.sqrt(D_sq)
                C_inv_sqrt = (B / D) @ B.T
            except np.linalg.LinAlgError:
                # Reset if unstable
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                C_inv_sqrt = np.eye(dim)
                sigma *= 0.5

            if sigma < 1e-8:
                break

        # Tail: Use remaining budget for directed random search using current distribution
        remaining = budget - calls
        if remaining > 0:
            for _ in range(remaining):
                z = np.random.standard_normal(dim)
                x_new = mean + sigma * (B * D) @ z
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls += 1
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
