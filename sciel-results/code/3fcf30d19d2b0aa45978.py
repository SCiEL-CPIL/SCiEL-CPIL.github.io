"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES style optimizer with adaptive step size for the Bent Cigar function.
    Uses a deadline guard to stay within the 300s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize at origin
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        sigma = 1.0  # step size
        C = np.eye(dim)  # covariance matrix
        p_sigma = np.zeros(dim)  # evolution path for sigma
        p_c = np.zeros(dim)  # evolution path for C
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        c_c = (4 + np.log(dim)) / (dim + 4)
        mu = lambda_ // 2  # number of recombination weights
        weights = np.log(np.arange(lambda_, 0, -1)) - np.log(lambda_)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))

        calls_made = 1
        for it in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            for i in range(lambda_):
                Y[i] = x + sigma * (C @ rng.standard_normal(dim))
                Y[i] = np.clip(Y[i], -5, 5)
                F[i] = func(Y[i])
                calls_made += 1
                if calls_made >= budget:
                    break

            if calls_made >= budget:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Update best
            if F[0] < best_fx:
                best_fx = F[0]
                best_x = Y[0].copy()

            # Recombination
            y_mean = np.zeros(dim)
            for i in range(mu):
                y_mean += weights[i] * Y[i]

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ (sigma * y_mean))
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (it + 1)) / chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1) * C
            for i in range(mu):
                C += c_mu * np.outer(Y[i], Y[i])

            # Update mean
            x = x + sigma * y_mean
            x = np.clip(x, -5, 5)

            # Adapt sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Check convergence
            if np.max(C) < 1e-15 or sigma < 1e-12:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
