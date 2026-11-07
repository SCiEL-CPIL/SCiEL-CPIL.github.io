"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 30s safety margin before 330s kill
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        popsize = 4 + int(3 * np.log(n))
        lambda_ = popsize
        c1 = (2 + 2 / (n + 1.3)) / ((n + 4) ** 2 + lambda_ + 2)
        cmu = (2 + 2 / (n + 1.3) - n * c1) / ((n + 4) + lambda_)
        cs = (lambda_ + 2) / (n + lambda_ + 5)
        ds = 1 + 2 * max(0, np.sqrt((lambda_ / (n + 1) - 1) / (n + 1)) - 1) + cs
        cc = (4 + n / lambda_) / (n + 4 + 2 * lambda_ / (n + 1))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial point and sigma
        x0 = rng.uniform(-5, 5, n)
        sigma = 2.0

        # CMA-ES state
        x_mean = x0.copy()
        B = np.eye(n)
        D = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        best_x = x0.copy()
        best_f = func(x0)
        calls_made = 1

        # Track best for AOCC
        for gen in range(budget // lambda_ + 1):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Generate population
            weights = np.ones(lambda_)
            mu = lambda_ // 2 + 1
            w = np.zeros(mu)
            w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1 / np.sum(w ** 2)

            for i in range(lambda_):
                if time.time() > deadline or calls_made >= budget:
                    break
                z = rng.standard_normal(n)
                y = x_mean + sigma * (B @ (D @ z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                if i < mu:
                    # Store for update
                    if i == 0:
                        ys = [y.copy()]
                        fs = [fy]
                    else:
                        ys.append(y.copy())
                        fs.append(fy)
            else:
                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = [ys[i] for i in order]
                fs_sorted = [fs[i] for i in order]

                # Weighted mean
                x_new_mean = np.sum([w[i] * ys_sorted[i] for i in range(mu)], axis=0)
                # CMA-ES update
                # Update covariance matrix
                for i in range(mu):
                    y_vec = (ys_sorted[i] - x_mean) / sigma
                    D = D + (w[i] / (2 * mu_eff)) * (np.outer(y_vec, y_vec) - np.eye(n))
                # Update p_sigma and p_c
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (D @ ((x_new_mean - x_mean) / sigma)))
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                p_c = (1 - cc) * p_c + h_sigma * cc * (x_new_mean - x_mean) / sigma
                # Update B and D
                for i in range(mu):
                    y_vec = (ys_sorted[i] - x_mean) / sigma
                    D = D + (w[i] / (2 * mu_eff)) * (np.outer(y_vec, y_vec) - np.eye(n))
                # Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(D)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs
                D = np.diag(np.sqrt(eigvals))
                x_mean = x_new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
