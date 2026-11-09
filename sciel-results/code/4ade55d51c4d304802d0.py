"""Optimizer for BBOB f7 (Step ellipsoidal) with CMA-ES."""
import numpy as np
import time


class Optimizer:
    """CMA-ES style population-based optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))  # population size
        # Initialize at center (minimizer is in [-4,4]^5, center 0 is good prior)
        x_mean = np.zeros(n)
        sigma = 1.5

        # Covariance matrix and evolution paths
        C = np.eye(n)
        C_eigenvalues = np.ones(n)
        C_eigenvectors = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # CMA-ES hyperparameters
        c_sigma = (lambda_ + 2) / (n + lambda_ + 5)
        c_c = (4 + lambda_ / n) / (n + 4 + 2 * lambda_ / n)
        c_mu = (2 + 2.0 / (n + lambda_ / 2)) / (n + lambda_ + c_mu + 2)
        mu = lam // 2
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(2 * (1 - c_1) / ((n + 2) ** 2 + mu_eff), 0.5 * c_1)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Initial population
        def evaluate(x):
            return func(np.clip(x, -5, 5))

        # Initial evaluation
        best_x = x_mean.copy()
        best_f = evaluate(best_x)
        calls = 1

        # Generate initial population
        def generate_population():
            nonlocal calls
            # Use Cholesky-like sampling from C
            # Simple: use eigendecomposition
            try:
                B = C_eigenvectors @ np.diag(np.sqrt(C_eigenvalues))
            except:
                B = np.eye(n)
            # Sample
            z = rng.standard_normal((lam, n))
            pop = x_mean + sigma * (z @ B.T)
            # Clip to box
            pop = np.clip(pop, -5, 5)
            # Evaluate
            fvals = np.array([evaluate(x) for x in pop])
            calls += lam
            return pop, fvals

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate and evaluate population
            pop, fvals = generate_population()

            # Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()

            # Weighted mean of top mu
            x_new_mean = np.sum(w[:, None] * pop[:mu], axis=0)

            # Update evolution paths
            y = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * 1))
                       < (1 + 2 / (n + 1)) * np.sqrt(mu_eff))
            p_c = ((1 - c_c) * p_c
                   + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            outer = np.outer(p_c, p_c)
            C = ((1 + c_1 * (c_1 + c_mu - 1) - delta_h) * C
                 + c_1 * outer
                 + c_mu / sigma ** 2 * np.sum(
                     w[i] * np.outer(pop[i] - x_mean, pop[i] - x_mean)
                     for i in range(mu)))

            # Update mean
            x_mean = x_new_mean

            # Re-estimate eigen decomposition periodically
            if calls % (lam * 5) < lam:
                try:
                    C_eigenvalues, C_eigenvectors = np.linalg.eigh(C)
                    C_eigenvalues = np.clip(C_eigenvalues, 1e-20, 1e10)
                except:
                    C = np.eye(n)
                    C_eigenvalues = np.ones(n)
                    C_eigenvectors = np.eye(n)

        return best_x, best_f
