"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

        # CMA-ES parameters
        N = max(4, self.dim)  # population size
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (N + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + c_sigma
        c_c = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        c_1 = 2 / ((N + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
        chi_N = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        x_mean = np.zeros(self.dim)
        sigma = 2.0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        best_x = None
        best_f = np.inf
        calls = 0

        def check_time():
            return time.time() > deadline

        # Initial evaluation
        fx = func(x_mean)
        calls += 1
        best_x = x_mean.copy()
        best_f = fx

        while calls < self.budget and not check_time():
            # Generate population
            pop = np.empty((N, self.dim))
            f_vals = np.empty(N)
            for i in range(N):
                if calls >= self.budget or check_time():
                    break
                z = rng.standard_normal(self.dim)
                # Sample from N(x_mean, sigma^2 * C)
                # Use Cholesky decomposition for C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    L = np.eye(self.dim)
                x_i = x_mean + sigma * (L @ z)
                x_i = np.clip(x_i, -5, 5)
                pop[i] = x_i
                f_i = func(x_i)
                calls += 1
                f_vals[i] = f_i
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()

            if calls >= self.budget or check_time():
                break

            # Sort by fitness
            order = np.argsort(f_vals)
            selected_indices = order[:mu]
            selected_pop = pop[selected_indices]
            selected_f = f_vals[selected_indices]

            # Weighted mean
            new_x_mean = np.sum(weights[:, None] * selected_pop, axis=0)

            # Update evolution paths
            y = (new_x_mean - x_mean) / sigma
            pc = (1 - c_c) * pc + c_c * y
            ps = (1 - c_sigma) * ps + np.sqrt(c_c * (2 - c_c) * mueff) * y

            # Update sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1)))) < (1.4 + 2 / (N + 1)) * chi_N
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1))

            # Update covariance matrix
            if hsig:
                delta_heav = c_c * (2 - c_c)
            else:
                delta_heav = 0
            C = (1 - c_1 - c_mu + delta_heav) * C + c_1 * (pc @ pc.T + delta_heav * C)
            for j in range(mu):
                x_j = (selected_pop[j] - x_mean) / sigma
                C += c_mu * weights[j] * np.outer(x_j, x_j)

            x_mean = new_x_mean

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(self.dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
