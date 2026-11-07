"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with covariance adaptation for rotated Weierstrass landscape."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s/330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        N = 4 + int(3 * np.log(dim))  # population size = 4 + 3*ln(5) ≈ 9
        mu = N // 2  # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 1.8) + c_sigma) / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        x_mean = np.zeros(dim)
        sigma = 2.5
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls_made = 0
        f_best = np.inf

        def evaluate(x):
            nonlocal calls_made, f_best
            calls_made += 1
            val = func(np.clip(x, -5, 5))
            if val < f_best:
                f_best = val
            return val

        # Initial evaluation
        fx = evaluate(x_mean)

        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            pop = np.array([
                x_mean + sigma * (B @ (D * rng.standard_normal(dim)))
                for _ in range(N)
            ])

            # Evaluate population
            pop_f = np.array([evaluate(x) for x in pop])

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_sorted = pop[order]
            f_sorted = pop_f[order]

            # Weighted mean of selected
            y_w = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

            # Step size control
            y_w_sigma = (B @ (D * (y_w / sigma)))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_w_sigma
            h_sigma = (
                np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * calls_made))
            ) / chi_N
            if h_sigma < 1.4:
                sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / chi_N - 1))

            # Covariance matrix adaptation
            y_c = y_w / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_c
            h_c = (
                np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * calls_made))
            ) / chi_N
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None] * (pop_sorted[:mu] - x_mean)[:, None] * (pop_sorted[:mu] - x_mean)[None, :], axis=0) / (sigma ** 2)

            C = (1 + c_1 * (h_c - 1) * c_c * (2 - c_c) + c_mu - c_1 - c_mu) * C
            C += c_1 * rank_one
            C += c_mu * rank_mu

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigenvalue decomposition for B and D
            try:
                eigenvalues, B_new = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-20)
                D = np.sqrt(eigenvalues)
                B = B_new
            except np.linalg.LinAlgError:
                B = np.eye(dim)
                D = np.ones(dim)
                C = np.eye(dim)

            # Update mean
            x_mean = x_mean + sigma * y_w
            x_mean = np.clip(x_mean, -5, 5)

        return x_mean, f_best
# EVOLVE-BLOCK-END
