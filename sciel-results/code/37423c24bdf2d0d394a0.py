"""Rewritten optimizer for BBOB-style f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial exploration: try several starting points
        candidates = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
        ]
        best_x = None
        best_f = np.inf
        calls_made = 0
        for c in candidates:
            c = np.clip(c, -5, 5)
            fc = func(c)
            calls_made += 1
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        # CMA-ES setup
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)

        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        mean = best_x.copy()
        C = np.eye(dim)
        # Cholesky of C
        C_chol = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        sigma = 1.0

        def update_chol(C, C_chol):
            """Update Cholesky factor from new C."""
            try:
                return np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                return np.linalg.cholesky(C + 1e-12 * np.eye(dim))

        while calls_made < budget and time.time() < deadline:
            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = C_chol @ z
                pop[i] = np.clip(mean + sigma * y, -5, 5)

            # Evaluate
            fvals = np.empty(lambda_)
            aborted = False
            for i in range(lambda_):
                if time.time() > deadline:
                    aborted = True
                    break
                fvals[i] = func(pop[i])
                calls_made += 1

            if aborted:
                break

            # Sort
            order = np.argsort(fvals)
            sorted_pop = pop[order]
            sorted_fvals = fvals[order]

            # Update best
            if sorted_fvals[0] < best_f:
                best_f = sorted_fvals[0]
                best_x = sorted_pop[0].copy()

            # Weighted mean
            w_mean = np.sum(w[:, None] * sorted_pop[:mu], axis=0)

            # Step size adaptation
            y_mean = (w_mean - mean) / sigma
            norm_y = np.linalg.norm(y_mean)
            h_sigma = min(1.0, np.sqrt(
                1 - (1 - c_sigma) ** (2 * (lambda_ - calls_made) / (dim + 3))
            ) if (lambda_ - calls_made) > 0 else 0)
            # Simpler h_sigma
            h_sigma = min(1.0, 1 - c_sigma) ** (1.0 / lambda_) if lambda_ > 0 else 0
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma * (norm_y / chi_n - 1) * y_mean
            sigma *= np.exp(c_sigma / d_sigma * (norm_y / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e3)

            # Save old mean for delta computation
            old_mean = mean.copy()
            mean = w_mean

            # Covariance matrix adaptation
            delta_mean = (w_mean - old_mean) / sigma
            y_w = delta_mean
            h_c = min(1.0, (1 - c_c) ** (2 * (lambda_ - calls_made) / (dim + 2)) + 1) if (lambda_ - calls_made) > 0 else 0
            h_c = min(1.0, (1 - c_c) ** (1.0 / lambda_)) if lambda_ > 0 else 0
            p_c = (1 - c_c) * p_c + c_c * h_c * y_w

            # Rank-1 and rank-mu updates
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (p_c @ p_c.T + (1 - h_c) * c_c * (2 - c_c) * np.eye(dim))
            for i in range(mu):
                y_i = (sorted_pop[i] - old_mean) / sigma
                C += c_mu * w[i] * np.outer(y_i - y_w, y_i - y_w)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update Cholesky
            C_chol = update_chol(C, C_chol)

        return best_x, best_f
# EVOLVE-BLOCK-END
