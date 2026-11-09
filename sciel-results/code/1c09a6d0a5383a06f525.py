"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 10, adaptive step size, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 330s kill

        # CMA-ES parameters
        lambda_ = 10  # population size
        mu = lambda_ // 2  # number of best solutions for recombination
        # Weights
        w = np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        # Step size adaptation (CSA)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Chi-squared expectation
        chi_lambda = np.sqrt(lambda_) * (1 - 1 / (4 * lambda_) + 1 / (21 * lambda_ ** 2))

        # Initialization
        rng = np.random.default_rng(42)
        x = rng.uniform(-4, 4, dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        # Evaluate initial point
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Precompute C^{1/2} initially
        def c_sqrt(C_mat):
            """Compute C^{1/2} via eigendecomposition."""
            evals, evecs = np.linalg.eigh(C_mat)
            evals = np.maximum(evals, 1e-20)
            return evecs @ np.diag(np.sqrt(evals)) @ evecs.T

        def c_inv_sqrt(C_mat):
            """Compute C^{-1/2} via eigendecomposition."""
            evals, evecs = np.linalg.eigh(C_mat)
            evals = np.maximum(evals, 1e-20)
            return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

        C_sqrt = c_sqrt(C)
        C_inv_sqrt = c_inv_sqrt(C)

        gen = 0
        while calls < budget and time.time() < deadline:
            gen += 1
            # Generate population
            Y = np.array([rng.standard_normal(dim) for _ in range(lambda_)])
            X = np.array([x + sigma * (C_sqrt @ y) for y in Y])
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(xi) for xi in X])
            calls += lambda_

            # Update best
            for i in range(lambda_):
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = X[i].copy()

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            X_sorted = X[order]
            F_sorted = F[order]

            # Mean recombination
            x_new = np.sum(w[:, None] * X_sorted[:mu], axis=0)

            # Update evolution path for covariance matrix adaptation
            pc = (1 - c_c) * pc + c_c * C_inv_sqrt @ (x_new - x) / sigma

            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_lambda - 1))
            sigma = max(sigma, 1e-12)

            # Update covariance matrix
            delta_heav = (1 + 2 * max(0, (mu_eff - 1) / (dim + 1))) * c_c
            outer = np.outer(pc, pc)
            C = (1 + c_1 + c_mu - c_c) * C + c_1 * (np.eye(dim) - C)
            C += c_mu * (np.sum(w[:, None, None] * np.outer(X_sorted[:mu] - x, X_sorted[:mu] - x) / sigma**2, axis=0) - C)
            C += c_c * (outer - C)
            # Symmetrize
            C = 0.5 * (C + C.T)

            # Ensure positive semi-definite
            evals, evecs = np.linalg.eigh(C)
            if np.any(evals < 1e-10):
                evals = np.maximum(evals, 1e-10)
                C = evecs @ np.diag(evals) @ evecs.T
                C = 0.5 * (C + C.T)

            x = x_new

            # Update C_sqrt and C_inv_sqrt periodically (every 5 generations for speed)
            if gen % 5 == 0:
                C_sqrt = c_sqrt(C)
                C_inv_sqrt = c_inv_sqrt(C)

            # Convergence check
            if sigma < 1e-10:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
