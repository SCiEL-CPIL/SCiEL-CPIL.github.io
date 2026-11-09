"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Full CMA-ES with covariance matrix adaptation for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
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

        # Initial random sampling
        n_init = min(10, budget)
        for _ in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            call_func(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # standard CMA-ES lambda
        mu = lam // 2  # number of weights
        # Weights
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Step size
        sigma = 1.0

        # Mean
        m = best_x.copy()

        # Covariance matrix (start as identity)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_cholesky = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Time constants
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))

        # Hessian update interval
        hsig = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)) / (dim + 2) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        generation = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            offspring_vals = []

            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = C_cholesky @ z
                x = m + sigma * y
                x = np.clip(x, -5.0, 5.0)
                fx = call_func(x)
                offspring.append((x, y, fx))

            if len(offspring) < mu:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[2])

            # Select best mu
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * offspring[i][1]

            # Update mean
            m_new = m + sigma * y_w
            m = m_new

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ y_w) / sigma
            hsig_ok = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (generation + 1)) < (1.4 + 2 / (dim + 1)) * chi_n

            p_c = (1 - c_c) * p_c + hsig_ok * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update covariance
            delta_hsig = (1 - hsig_ok) * np.sqrt(c_c * (2 - c_c) * mu_eff) * chi_n
            C = (1 + c_1 * (c_mu + delta_hsig) - c_1 - c_mu) * C
            for i in range(mu):
                y_i = offspring[i][1]
                C += c_1 * np.outer(y_i, y_i) + c_mu * w[i] * np.outer(y_i, y_i)

            # Regularize C to be positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, 1e10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update Cholesky
            try:
                C_cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_cholesky = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            generation += 1

            # Stagnation check: if no improvement in last 30 generations, restart
            if generation % 30 == 0:
                recent_best = min(fx for _, _, fx in offspring[:mu])
                if recent_best >= best_f * 1.001:
                    # Try a random restart
                    for _ in range(2):
                        if calls >= budget or time.time() > deadline:
                            break
                        new_x = rng.uniform(-4, 4, dim)
                        new_fx = call_func(new_x)
                        if new_fx < best_f * 0.99:
                            m = new_x.copy()
                            sigma = 1.0
                            C = np.eye(dim)
                            C_cholesky = np.eye(dim)
                            p_sigma = np.zeros(dim)
                            p_c = np.zeros(dim)
                            generation = 0
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
