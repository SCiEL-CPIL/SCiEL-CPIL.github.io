"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive covariance matrix for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial population
        x_mean = np.zeros(dim)
        sigma = 2.0
        pop = rng.normal(0, 1, (lambda_, dim))
        pop = x_mean + sigma * pop
        f_vals = np.array([call_func(pop[i]) for i in range(lambda_)])

        # Initial best
        idx_best = np.argmin(f_vals)
        best_x = pop[idx_best].copy()
        best_f = f_vals[idx_best]

        # CMA-ES state
        ps = np.zeros(dim)
        hs = np.zeros(dim)
        C = np.eye(dim)
        Cs = np.eye(dim)
        dcor = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_c

        while calls < budget:
            if time.time() > deadline:
                break

            # Sort population by fitness
            order = np.argsort(f_vals)
            pop = pop[order]
            f_vals = f_vals[order]

            # Update mean
            x_new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update sigma control
            z = (x_new_mean - x_mean) / (sigma + 1e-30)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lambda_ + 1)))) < (1 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance matrix
            if hsig:
                ps = (1 - c_sigma) * ps
            Cs = (1 - c_c) * Cs + c_c * (np.outer(ps, ps) - Cs)
            v = np.sum(weights[:, None] * pop[:mu], axis=0)
            diff = pop[:mu] - x_mean
            rank_mu = np.sum(weights[:, None] * np.outer(diff, diff), axis=0)
            C = (1 + c_1 * (hsig - 1) - c_1 - c_mu) * C + c_1 * (Cs - C) + c_mu * rank_mu

            # Ensure C is symmetric positive definite
            C = 0.5 * (C + C.T)
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-10, None)
                C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            except np.linalg.LinAlgError:
                C = np.eye(dim) * sigma

            # Update mean
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5.0, 5.0)

            # Generate new population
            pop = np.zeros((lambda_, dim))
            f_vals = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline or calls >= budget:
                    break
                # Sample from multivariate normal
                z = rng.normal(0, 1, dim)
                # Transform to get correlated normal
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ z
                except np.linalg.LinAlgError:
                    y = z
                pop[i] = x_mean + sigma * y
                pop[i] = np.clip(pop[i], -5.0, 5.0)
                f_vals[i] = call_func(pop[i])

            # Update global best
            idx_best = np.argmin(f_vals)
            if f_vals[idx_best] < best_f:
                best_f = f_vals[idx_best]
                best_x = pop[idx_best].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
