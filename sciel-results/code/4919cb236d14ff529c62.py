"""CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population-based search and adaptive covariance."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        # --- CMA-ES parameters ---
        # Population size: lambda = 4 + floor(3 * ln(dim))
        lam = 4 + int(3 * np.log(max(dim, 2)))
        # Step size initialization
        sigma = 2.0
        # Initial mean: center of box
        x_mean = np.zeros(dim)
        # Initial covariance: identity
        C = np.eye(dim)
        # Eigen decomposition cache
        D = np.ones(dim)  # diagonal of C = 1/sqrt(eigenvalues)
        B = np.eye(dim)   # eigenvectors
        # Adaptation parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls_made += 1
            return fx

        # Initial evaluation
        f_best = eval_func(x_mean)
        best_x = x_mean.copy()
        best_f = f_best

        # Main loop
        while calls_made < budget and time.time() < deadline:
            # Generate population
            ys = []
            for i in range(lam):
                z = np.random.randn(dim)
                y = x_mean + sigma * (B @ (D * z))
                ys.append(y)
            ys = np.array(ys)

            # Evaluate population
            fvals = np.array([eval_func(y) for y in ys])
            if calls_made >= budget or time.time() >= deadline:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = ys[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = ys_sorted[0].copy()

            # Update mean (weighted recombination)
            x_new = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)

            # Update step-size control (CSA)
            y_old = (x_mean - x_new) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_old
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_made + 1)) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Update covariance
            y_new = (x_new - x_mean) / sigma
            delta_sigma = np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1)) - 1
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_new
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                yi = (ys_sorted[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(yi, yi)
            C = (1 + c_c * (2 - c_c) * delta_sigma - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                B, D_inv = np.linalg.eigh(C)
                D = 1.0 / np.sqrt(np.maximum(D_inv, 1e-20))
            except np.linalg.LinAlgError:
                B = np.eye(dim)
                D = np.ones(dim)

            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
