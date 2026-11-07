"""CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pure CMA-ES with population size 10 for the attractive sector."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # 15s safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # CMA-ES parameters
        pop_size = 10
        lambda_ = pop_size
        mu = lambda_ // 2  # number of weights

        # Weights for the mean update
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size initialization
        sigma = 2.0

        # Initial mean
        x_mean = np.zeros(dim)

        # Covariance matrix (identity initially)
        C = np.eye(dim)

        # Eigen decomposition for sampling
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-20))

        # Parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Generation counter
        gen = 0

        # Initial evaluation of mean
        f_mean = func(x_mean)
        calls_made += 1
        best_f = f_mean
        best_x = x_mean.copy()

        # CMA-ES main loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate population
            population = np.zeros((lambda_, dim))
            fvals = np.zeros(lambda_)

            for i in range(lambda_):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Standard normal
                z = np.random.standard_normal(dim)
                # Transform and sample
                x = x_mean + sigma * (B @ (D * z))
                x = np.clip(x, -5, 5)

                population[i] = x
                fvals[i] = func(x)
                calls_made += 1

                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x.copy()

            if calls_made >= budget:
                break

            # Sort by function values
            order = np.argsort(fvals)
            fvals_sorted = fvals[order]
            x_sorted = population[order]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = x_sorted[0].copy()

            # Update mean
            x_old = x_mean.copy()
            for i in range(mu):
                x_mean += weights[i] * x_sorted[i]

            # Update step size
            y_mean = (x_mean - x_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 1e8)

            # Update covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_mean

            h_sigma = int(
                (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))))
                < (1.4 + 2 / (dim + 1)) * chi_n
            )

            C = (1 + c_1 * h_sigma * (dim + 2) - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) - h_sigma * C)
            for i in range(mu):
                y_i = (x_sorted[i] - x_old) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))

            gen += 1

            # Convergence check
            if np.linalg.norm(x_mean - x_old) < 1e-10:
                break

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            if calls_made < budget:
                best_f = func(x)
                calls_made += 1
            else:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
