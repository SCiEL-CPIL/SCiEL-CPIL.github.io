"""CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population-based sampling, covariance adaptation, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # 15s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        calls_made = 0

        # CMA-ES parameters
        pop_size = 10  # lambda
        mu = pop_size // 2  # number of recombination weights
        # Logarithmic weighting
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size initialization
        sigma = 2.0

        # Mean vector - start at origin
        m = np.zeros(dim)

        # Covariance matrix - identity
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Track global best
        best_x = None
        best_f = np.inf

        # Evaluate initial mean
        fx = func(m)
        calls_made += 1
        best_f = fx
        best_x = m.copy()

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate population
            population = []
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sample from N(m, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Cholesky decomposition of C for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

                x_i = m + sigma * L @ z
                x_i = np.clip(x_i, -5, 5)

                fx_i = func(x_i)
                calls_made += 1

                if fx_i < best_f:
                    best_f = fx_i
                    best_x = x_i.copy()

                population.append((x_i, fx_i))

            if len(population) < pop_size:
                break

            # Sort by fitness
            population.sort(key=lambda x: x[1])

            # Recombination
            m_new = np.sum([weights[i] * population[i][0] for i in range(mu)], axis=0)

            # Update sigma (CSA)
            old_sigma = sigma
            old_C_sqrt = C_sqrt.copy()
            C_inv_sqrt = np.linalg.inv(C_sqrt)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ ((m_new - m) / (sigma * old_sigma))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))

            # Update mean
            m = m_new

            # Update covariance
            old_C = C.copy()
            delta = (m_new - m) / sigma if sigma > 0 else np.zeros(dim)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ delta

            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = (population[i][0] - m_new) / (sigma * old_sigma) if sigma * old_sigma > 0 else np.zeros(dim)
                rank_mu += weights[i] * np.outer(diff, diff)

            C = (1 + c_1 + c_mu - c_1 / mu_eff) * old_C + c_1 * (rank_one - (1 - c_c) * (1 - np.trace(C_inv @ rank_one) / dim) * old_C) + c_mu * rank_mu

            # Regularize C to maintain positive semi-definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2  # Ensure symmetry

            # Update C_inv and C_sqrt
            try:
                C_inv = np.linalg.inv(C)
                C_inv = (C_inv + C_inv.T) / 2
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if Cholesky fails
                C_inv = np.eye(dim)
                C_sqrt = np.eye(dim)
                C = np.eye(dim)

            # Check for convergence
            if sigma < 1e-10:
                break

            # Check if progress is too slow - restart with new center
            if calls_made > budget - 50:
                break

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            best_f = func(x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
