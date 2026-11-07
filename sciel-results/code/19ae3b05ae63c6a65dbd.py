"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        calls_made = 1
        x_best = x0.copy()
        f_best = f0

        # CMA-ES parameters
        lambda_ = 4 * int(np.ceil(3 + np.log(dim)))  # population size
        mu = lambda_ // 2  # number of selected individuals
        # Weights for selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)  # effective population size

        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Step size
        sigma = 2.0
        # Mean
        m = x0.copy()
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C
        C_sqrt_inv = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Chi-squared correction
        chi_lambda = np.sqrt(lambda_) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Helper function to update C_sqrt_inv
        def update_C_sqrt_inv(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            return C_sqrt_inv

        # Main CMA-ES loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                Y[i] = y
                F[i] = fy

            if calls_made < budget:
                # Sort by fitness
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = F[order]

                # Update best
                if F_sorted[0] < f_best:
                    f_best = F_sorted[0]
                    x_best = Y_sorted[0].copy()

                # Compute weighted mean
                y_w = np.sum(weights[:, np.newaxis] * Y_sorted[:mu], axis=0)
                y_w = np.clip(y_w, -5, 5)

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt_inv @ (y_w - m) / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - lambda_))) <
                        (1 + 2 / (dim + 1)) * chi_lambda)
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (y_w - m) / sigma

                # Update mean
                m = y_w

                # Update covariance matrix
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                rank_one = np.outer(pc, pc)
                rank_mu = np.sum(weights[:, np.newaxis] *
                                 np.outer(Y_sorted[:mu] - m, Y_sorted[:mu] - m), axis=0) / sigma ** 2
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * (rank_one + delta_hsig * np.outer(pc, pc)) + cmu * rank_mu

                # Symmetrize C
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(cs / chi_lambda * (np.linalg.norm(ps) / chi_lambda - 1))

                # Update C_sqrt_inv
                C_sqrt_inv = update_C_sqrt_inv(C)

        return x_best, f_best
# EVOLVE-BLOCK-END
