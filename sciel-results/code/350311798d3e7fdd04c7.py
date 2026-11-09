"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        budget = self.budget
        dim = self.dim

        calls = 0
        best_fx = np.inf
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_fx, best_x
            if calls >= budget:
                return np.inf
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            if np.isfinite(fx) and fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # CMA-ES parameters
        # Population size
        lam = int(4 + 3 * np.log(dim))
        # Selection
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Initial mean
        x_mean = np.zeros(dim)
        # Initial sigma
        sigma = 1.0

        # Evolution path
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        # Precompute sqrt of C using Cholesky
        try:
            B = np.linalg.cholesky(C)
            D = np.diag(1.0 / np.sqrt(np.diag(C)))
            C_mat = np.eye(dim)
        except:
            B = np.eye(dim)
            D = np.eye(dim)
            C_mat = np.eye(dim)

        # Restart conditions
        restarts = 0
        max_restarts = 2

        while calls < budget and time.time() < deadline and restarts <= max_restarts:
            # Reset if restarting
            if restarts > 0:
                x_mean = best_x.copy()
                sigma = 0.5
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                C = np.eye(dim)
                try:
                    B = np.linalg.cholesky(C)
                    D = np.diag(1.0 / np.sqrt(np.diag(C)))
                    C_mat = np.eye(dim)
                except:
                    B = np.eye(dim)
                    D = np.eye(dim)
                    C_mat = np.eye(dim)

            # Main loop
            while calls < budget and time.time() < deadline:
                # Sample population
                pop = []
                f_vals = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    # Transform z using C^(1/2)
                    y = B.T @ (D @ (B @ z))
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = safe_func(x_new)
                    pop.append(x_new)
                    f_vals.append(f_new)

                if calls >= budget:
                    break

                # Sort
                order = np.argsort(f_vals)
                pop_sorted = [pop[i] for i in order]
                f_vals_sorted = [f_vals[i] for i in order]

                # Check convergence
                if f_vals_sorted[-1] - f_vals_sorted[0] < 1e-12 * max(1, abs(f_vals_sorted[0])):
                    break

                # Update paths and covariance
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * (pop_sorted[i] - x_mean) / sigma

                # Update sigma evolution path
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update C evolution path
                y_c = y_w
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

                # Rank-one and rank-mu updates
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    diff = (pop_sorted[i] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(diff, diff)

                C = (1 + c_1 * (np.trace(C) / dim - 1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update mean
                x_mean = x_mean + sigma * y_w
                x_mean = np.clip(x_mean, -5.0, 5.0)

                # Update Cholesky factorization
                try:
                    B = np.linalg.cholesky(C)
                    D = np.diag(1.0 / np.sqrt(np.diag(C)))
                except:
                    # If C is not positive definite, reset
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.eye(dim)
                    p_c = np.zeros(dim)
                    p_sigma = np.zeros(dim)
                    sigma = 0.5
                    x_mean = best_x.copy()
                    restarts += 1
                    break

                # Check for stagnation
                if sigma < 1e-8 or np.all(np.abs(f_vals_sorted[:mu] - f_vals_sorted[0]) < 1e-10):
                    break

            # If we exited due to stagnation, restart
            if calls < budget and time.time() < deadline and restarts <= max_restarts:
                if sigma < 1e-8 or np.all(np.abs(f_vals_sorted[:mu] - f_vals_sorted[0]) < 1e-10):
                    restarts += 1
                    if restarts > max_restarts:
                        break

        # Final Polish with (1+1)-ES if budget remains
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.01

            while calls < budget and time.time() < deadline:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9

                if sigma < 1e-8:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
