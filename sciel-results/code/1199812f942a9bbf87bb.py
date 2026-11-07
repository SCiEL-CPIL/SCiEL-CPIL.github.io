"""CMA-ES based optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive sigma for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        # Start at center of the box
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size

        # Covariance matrix (start with identity)
        C = np.eye(dim)
        # Eigendecomposition of C
        # C = B @ diag(D^2) @ B.T
        # We need B and D (sqrt of eigenvalues)
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Population size
        # Standard CMA-ES: lambda = 4 + 3*ln(n)
        # For n=5, lambda = 4 + 3*ln(5) = 4 + 4.82 = 8.82 -> 9
        lambda_ = 4 + int(3 * np.log(dim))
        if lambda_ < 2:
            lambda_ = 2
        
        mu = lambda_ // 2  # Number of recombination weights

        # Recombination weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w ** 2).sum()

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Threshold for h_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Evolution path for sigma
        p_sigma = np.zeros(dim)
        # Evolution path for covariance
        p_c = np.zeros(dim)

        # Initial function evaluation
        fx = func(mean)
        best_x = mean.copy()
        best_fx = fx
        calls = 1

        # Precompute C_inv_sqrt = B @ diag(1/D) @ B.T
        C_inv_sqrt = np.eye(dim)

        # Generation counter for h_sigma threshold
        gen = 0

        # Main CMA-ES loop
        while calls + lambda_ <= budget:
            # Generate population
            Z = np.random.standard_normal((lambda_, dim))
            Y = (B * D) @ Z.T
            X = mean + sigma * Y.T
            X = np.clip(X, -5, 5)

            # Evaluate population
            F = np.array([func(x) for x in X])
            calls += lambda_
            gen += 1

            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_fx:
                best_fx = F[min_idx]
                best_x = X[min_idx].copy()

            # Sort by fitness
            idx = np.argsort(F)
            F_sorted = F[idx]
            X_sorted = X[idx]

            # Update mean
            mean_old = mean.copy()
            mean = np.sum(w[:, None] * X_sorted[:mu], axis=0)

            # Update evolution paths
            diff = (mean - mean_old) / sigma
            
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff)
            
            # h_sigma with proper time-dependent threshold
            # ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*gen)) < (1+2/(n+1))*chi_n
            decay_factor = 1 - (1 - c_sigma) ** (2 * gen)
            if decay_factor < 1e-12:
                h_sigma = 1.0
            else:
                threshold = (1 + 2 / (dim + 1)) * chi_n * np.sqrt(decay_factor)
                h_sigma = 1.0 if np.linalg.norm(p_sigma) < threshold else 0.0

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)

            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff)

            # Update covariance matrix
            # Y_selected is the normalized steps for the mu best points
            Y_selected = (X_sorted[:mu] - mean_old) / sigma
            
            # Update C with rank-1 and rank-mu updates
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Rank-mu update: sum(w_i * outer(y_i, y_i))
            # Y_selected has shape (mu, dim), w has shape (mu,)
            # We need sum_i w_i * y_i^T y_i = Y_selected.T @ (w[:, None] * Y_selected)
            C += c_mu * (Y_selected.T @ (w[:, None] * Y_selected))

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigendecomposition
            try:
                D_sq, B = np.linalg.eigh(C)
                # Handle numerical issues
                D_sq = np.maximum(D_sq, 1e-12)
                D = np.sqrt(D_sq)
                C_inv_sqrt = (B / D) @ B.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                C_inv_sqrt = np.eye(dim)
                sigma *= 0.5

            # Check for convergence
            if sigma < 1e-8:
                break

        # Use remaining budget for local search
        remaining = budget - calls
        if remaining > 0:
            # Adaptive local search with coordinate descent mixed with random steps
            local_sigma = max(sigma, 1e-3)
            
            for i in range(remaining):
                # Mix of coordinate descent and random steps
                if i % dim == 0 and remaining - i >= dim:
                    # Try coordinate descent: one dimension at a time
                    # Pick the dimension with largest step
                    dim_idx = i % dim
                    step = local_sigma * np.random.choice([-1, 1])
                    x_new = best_x.copy()
                    x_new[dim_idx] += step
                    x_new = np.clip(x_new, -5, 5)
                    fx_new = func(x_new)
                    calls += 1
                    if fx_new < best_fx:
                        best_fx = fx_new
                        best_x = x_new.copy()
                        local_sigma *= 1.2
                    else:
                        local_sigma *= 0.8
                else:
                    # Random step
                    z = np.random.standard_normal(dim)
                    x_new = best_x + local_sigma * z
                    x_new = np.clip(x_new, -5, 5)
                    fx_new = func(x_new)
                    calls += 1
                    if fx_new < best_fx:
                        best_fx = fx_new
                        best_x = x_new.copy()
                        local_sigma *= 1.1
                    else:
                        local_sigma *= 0.9
                
                # Ensure local_sigma stays in reasonable range
                local_sigma = np.clip(local_sigma, 1e-6, 10.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
