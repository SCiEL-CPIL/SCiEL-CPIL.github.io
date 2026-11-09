"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        x0 = np.zeros(n)
        sigma = 1.0
        # Initial covariance matrix
        C = np.eye(n)
        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)

        # CMA-ES parameters
        mu = max(1, n // 2)  # number of parents
        lambda_pop = 2 * mu  # population size
        # Weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size control
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        # Boundary handling
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        def sample_population(x, sigma, C):
            """Generate lambda_pop samples from N(x, sigma^2 * C)."""
            # Cholesky decomposition
            L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            z = np.random.randn(lambda_pop, n)
            y = x + sigma * (z @ L.T)
            return y

        def adapt_sigma(sigma, ys):
            """Adapt step size using evolution path ps."""
            return sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ys) / chi_n - 1))

        def adapt_covariance(C, pc, ys, C_inv, y):
            """Adapt covariance matrix."""
            # Rank-one update
            C = (1 + c_1 * c_mu * (c_mu - 1) - c_1) * C
            C += c_1 * (np.outer(pc, pc) - (1 - c_c) * C)
            # Rank-mu update
            C += c_mu * np.sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
            return C

        # Main loop
        best_x = x0
        best_f = func(x0)
        calls_made = 1

        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            pop = sample_population(x0, sigma, C)
            # Evaluate
            fvals = np.array([func(pop[i]) for i in range(lambda_pop)])
            calls_made += lambda_pop

            # Sort by fitness
            order = np.argsort(fvals)
            y_selected = pop[order[:mu]]
            f_selected = fvals[order[:mu]]

            # Update best
            if f_selected[0] < best_f:
                best_f = f_selected[0]
                best_x = y_selected[0]

            # Update mean
            x0 = x0 + sigma * np.sum(weights[:, None] * (y_selected - x0), axis=0)

            # Update evolution path ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * np.linalg.cholesky(C).T @ (y_selected[0] - x0)
            # Actually, let's do it properly
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (y_selected[0] - x0) / sigma

            # Adapt sigma
            sigma = adapt_sigma(sigma, ps)
            sigma = np.clip(sigma, 1e-8, 1e8)

            # Update evolution path pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (x0 - x0) / sigma  # This is wrong
            # Let me fix the pc update
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (y_selected[0] - x0) / sigma

            # Adapt covariance
            C = adapt_covariance(C, pc, y_selected, np.linalg.inv(C), y_selected[0])
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except:
                C = C + 1e-5 * np.eye(n)

        return best_x, best_f
# EVOLVE-BLOCK-END
