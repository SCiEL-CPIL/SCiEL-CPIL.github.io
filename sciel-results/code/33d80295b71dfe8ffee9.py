# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts and bounded sampling for BBOB f18."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 260.0

        calls = 0
        best_x = None
        best_f = np.inf

        def eval_pt(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        # Initial settings
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Update rules
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1) + c_sigma)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim**2))

        best_f_run = np.inf
        no_improve_count = 0
        restart_count = 0
        max_no_improve = 30

        # Evaluate initial mean
        f_mean = eval_pt(x_mean)
        if f_mean < best_f_run:
            best_f_run = f_mean

        while calls < budget and time.time() < deadline:
            # Check for restart
            if no_improve_count > max_no_improve or sigma < 1e-10 or sigma > 1e4:
                if restart_count == 0:
                    x_mean = np.random.uniform(-4.0, 4.0, dim)
                    sigma = 2.0
                else:
                    if best_x is not None:
                        scale = max(0.5, sigma * np.sqrt(dim) * 0.5)
                        x_mean = best_x + np.random.normal(0, scale, dim)
                        x_mean = np.clip(x_mean, -5.0, 5.0)
                    else:
                        x_mean = np.zeros(dim)
                    sigma = max(1.0, sigma * 0.5)
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0
                restart_count += 1
                best_f_run = best_f
                continue

            # Sample population
            X = []
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                x_i = x_mean + sigma * (L @ z)
                X.append(x_i)

            # Evaluate population
            F = np.array([eval_pt(x_i) for x_i in X])

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = [X[i] for i in order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_f_run - 1e-12:
                best_f_run = F_sorted[0]
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Update mean
            x_new_mean = np.sum(weights[:, None] * np.array(X_sorted[:mu]), axis=0)

            # Update covariance matrix
            diff = np.array(X_sorted[:mu]) - x_mean
            C = (1 - c_c * (2 - c_c)) * C + c_c * (2 - c_c) * (x_new_mean - x_mean)[:, None] * (x_new_mean - x_mean)[None, :]
            rank_mu_update = np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            C = C + c_c * (2 - c_c - c_sigma * (1 - d_sigma)) * rank_mu_update

            # Ensure positive semi-definite
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                C = C + 1e-10 * np.eye(dim)
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)

            # Update paths
            sigma_old = sigma
            try:
                C_inv = np.linalg.inv(C)
                w, V = np.linalg.eigh(C_inv)
                w = np.clip(w, 0, None)
                sqrt_C_inv = V @ np.diag(np.sqrt(w)) @ V.T
            except np.linalg.LinAlgError:
                sqrt_C_inv = np.eye(dim)

            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e4)

            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt_C_inv @ (x_new_mean - x_mean) / sigma_old
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma_old

            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5.0, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
