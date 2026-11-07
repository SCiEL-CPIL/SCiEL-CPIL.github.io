"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for BBOB f11 (Discus) with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # CMA-ES parameters
        sigma = 2.0
        # Covariance matrix C (start as identity)
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Step sizes for paths
        c_sigma = (4 + dim / 2.0) / (dim + 4 + 2.0 * dim / dim)
        c_sigma = 1.0 / (dim + 3)  # standard
        c_c = (4 + dim / dim) / (dim + 4 + 3 * dim / dim)
        c_c = 1.0 / (dim + 3)  # simplified
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (dim + 2) - 1))
        mu_eff = dim / 2.0  # effective population size

        # Population size
        lambda_pop = 4 * int(dim / 2.0) + 2
        mu = lambda_pop // 2

        # Eigen decomposition for C
        eigenvalues = np.ones(dim)

        for _ in range(budget - calls):
            if time.time() > deadline:
                break

            # Generate population
            # Use Cholesky-like approach via eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                B = eigvecs @ np.diag(np.sqrt(eigvals))
            except:
                B = np.eye(dim)

            # Generate lambda_pop candidates
            Y = rng.standard_normal((lambda_pop, dim))
            X = np.tile(x, (lambda_pop, 1)) + sigma * (Y @ B.T)
            X = np.clip(X, -5, 5)

            # Evaluate
            F = []
            for i in range(lambda_pop):
                fi = func(X[i])
                F.append(fi)
                calls += 1
                if fi < best_fx:
                    best_fx = fi
                    best_x = X[i].copy()
                if time.time() > deadline:
                    # Finish evaluating remaining with best guess
                    F.extend([best_fx] * (lambda_pop - i - 1))
                    break

            if len(F) < lambda_pop:
                break

            F = np.array(F)
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_fx:
                best_fx = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Weighted mean of best mu
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()

            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

            # Update evolution paths
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (sigma * (m_new - x) / sigma))
            # Actually, the standard formula uses the normalized step
            # Let me use a simpler but effective update

            # p_c update
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls))) < (1 + 2 / (dim + 1)) * np.sqrt(c_sigma * (2 - c_sigma))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - x) / sigma

            # Update x
            x = m_new

            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-8, 50.0)

            # Update C
            delta_d = (1 / c_c + 2 * mu_eff) - 1 + c_c * h_sigma * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None] * np.outer(X_sorted[:mu] - x, X_sorted[:mu] - x), axis=0)
            C = (1 + c_1 * delta_d - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = (eigvecs * eigvals) @ eigvecs.T
            except:
                C = np.eye(dim)

        return best_x, best_fx
# EVOLVE-BLOCK-END
