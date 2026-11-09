"""Rewritten program using a population-based CMA-ES for the Rastrigin rotated task."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based CMA-ES with lambda=10 for global diversity."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        # Start at the center of the box
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size

        # Population size
        # For dim=5, lambda is typically 4 + floor(3*log(dim + 1))
        # But the directive suggests lambda=10 for large initial population
        lambda_ = 10
        mu = lambda_ // 2  # Number of parents

        # Weights for recombination
        w = np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        # Initial evaluation
        x0 = mean.copy()
        fx = func(x0)
        best_x = x0.copy()
        best_f = fx
        calls = 1

        # Pre-compute constants
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while calls < budget:
            # Generate population
            pop = []
            for i in range(lambda_):
                # Sample from multivariate normal
                # Use Cholesky decomposition of C
                # For efficiency, we can use the fact that C is symmetric positive definite
                # However, for dim=5, we can just do it directly
                # But we need to be careful with the budget.
                # Generating the population takes 0 function calls, but sampling does.
                # We need to ensure we don't exceed the budget.
                # Each individual in the population requires 1 function call.
                # So we can only generate min(lambda_, budget - calls) individuals.
                pass

            # Determine how many individuals we can evaluate in this generation
            n_eval = min(lambda_, budget - calls)
            if n_eval == 0:
                break

            # Generate n_eval samples
            # We need to sample from N(mean, sigma^2 * C)
            # We can use scipy.stats.multivariate_normal or manual Cholesky
            # To avoid scipy dependency issues or overhead, let's do manual
            # But the prompt says SciPy is available. Let's use it for robustness.
            # Actually, manual is safer for performance and dependency control.
            # Let's use a simple Cholesky update or just eigendecomposition if C changes often.
            # For dim=5, eigendecomposition is cheap.

            # Update C and sigma based on previous generation?
            # No, C and sigma are updated after evaluating the population.
            # So for the first generation, C=I, sigma=2.

            # Sample
            # We need to sample from N(mean, sigma^2 * C)
            # Let's compute the square root of C.
            # Since C is symmetric, we can use eigendecomposition.
            # Or we can use Cholesky if C is positive definite.
            # CMA-ES maintains C as positive definite.
            # Let's use Cholesky.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigendecomposition if Cholesky fails
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 0)
                L = eigvecs @ np.diag(np.sqrt(eigvals))

            # Generate standard normal samples
            Z = np.random.default_rng(0).standard_normal((n_eval, dim))
            # Transform to desired distribution
            # x = mean + sigma * (Z @ L.T)
            # Note: Z is (n_eval, dim), L is (dim, dim)
            # Z @ L.T gives (n_eval, dim)
            X = mean + sigma * (Z @ L.T)

            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate
            new_xs = []
            new_fs = []
            for i in range(n_eval):
                x_i = X[i]
                f_i = func(x_i)
                new_xs.append(x_i)
                new_fs.append(f_i)
                calls += 1

                # Update best
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()

            # Sort by fitness
            idx = np.argsort(new_fs)
            new_xs = np.array(new_xs)
            new_fs = np.array(new_fs)
            new_xs = new_xs[idx]
            new_fs = new_fs[idx]

            # Update mean
            # y_w = sum(w_i * (x_i - mean)) / sigma
            y_w = np.sum(w[:n_eval, np.newaxis] * (new_xs[:n_eval] - mean)[np.newaxis, :, :], axis=0) / sigma

            # Update path evolution strategies
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_w / np.linalg.norm(y_w) if np.linalg.norm(y_w) > 1e-12 else 0)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update covariance matrix
            # Delta h term
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls))) < (1 + 2 / (dim + 1)) * chi_n else 0.0
            # Note: The condition number depends on the number of generations, not calls.
            # Let's simplify the h_sigma update for now, or use a standard approximation.
            # A common approximation is to check if the norm is within expected range.

            # Rank-one update
            C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T - (1 - h_sigma) * C)
            # Rank-mu update
            C += c_mu * np.sum(w[:n_eval, np.newaxis, np.newaxis] * (new_xs[:n_eval] - mean)[:, np.newaxis, :] @ (new_xs[:n_eval] - mean)[:, :, np.newaxis], axis=0)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update mean
            mean = mean + sigma * y_w

            # Clip mean to box? Not strictly necessary, but good practice
            # mean = np.clip(mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
