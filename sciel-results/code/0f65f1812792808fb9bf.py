"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """IPOP-CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 280.0  # 20s safety margin
        rng = np.random.default_rng(12345)

        # Initial point: center of the box (0,0,...,0) is a reasonable start
        x = np.zeros(self.dim)
        fx = func(x)
        self.calls += 1
        self.best_x = x.copy()
        self.best_f = fx

        # IPOP-CMA-ES parameters
        n = self.dim
        # Initial population size
        lambda_ = int(4 + 3 * np.log(n))
        # Initial sigma
        sigma = 1.0

        # CMA-ES state
        x_mean = x.copy()
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)

        # Step sizes for Hessian-free updates
        mu = lambda_ // 2
        w = np.full(mu, 1.0)
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 9 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        count_eval = 1

        while self.calls < self.budget:
            # Check deadline
            if time.time() > self.deadline:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(lambda_):
                # Check deadline inside loop too
                if time.time() > self.deadline or self.calls >= self.budget:
                    break
                z = rng.standard_normal(n)
                y = x_mean + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                self.calls += 1
                ys.append(y)
                fs.append(fy)
                if fy < self.best_f:
                    self.best_f = fy
                    self.best_x = y.copy()
                count_eval += 1

            if len(ys) < 2:
                break

            ys = np.array(ys)
            fs = np.array(fs)

            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]

            # Update mean
            x_mean_new = np.sum(w[:, None] * ys[:mu], axis=0)

            # Update covariance and step size
            if count_eval % 2 == 0:  # Only update every 2 evaluations to save time
                y_sigma = (x_mean_new - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) /
                           np.sqrt(1 - (1 - c_sigma) ** (count_eval / d_sigma)) /
                           np.sqrt(1 - (1 - c_sigma) ** (count_eval / d_sigma)) if False else
                           (np.linalg.norm(p_sigma) /
                            np.sqrt(1 - (1 - c_sigma) ** (count_eval / d_sigma))))
                delta_sigma = 1.0 - h_sigma < 1.449 + 2 / (n + 1)
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) /
                                                           np.sqrt(1 - (1 - c_sigma) ** (count_eval / d_sigma)) - 1))

                p_c = (1 - c_c) * p_c + delta_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma

                # Covariance update
                C = ((1 - c_1 - c_mu) * C +
                     c_1 * (np.outer(p_c, p_c) + (1 - delta_sigma * c_c * (2 - c_c)) * C) +
                     c_mu * np.sum(w[:, None, None] * np.outer(ys[:mu] - x_mean, ys[:mu] - x_mean), axis=0))

                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update square roots
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))

            x_mean = x_mean_new

            # IPOP: reduce population size if stagnating
            if count_eval % 10 == 0 and self.calls < self.budget:
                # Check for stagnation
                if len(fs) > mu and np.std(fs[:mu]) < 1e-6:
                    lambda_ = max(int(lambda_ * 0.5), 4)
                    sigma *= 2.0

            # Reset if stuck
            if self.calls > 100 and self.best_f > 0 and count_eval % 50 == 0:
                # Restart with larger sigma
                sigma = min(sigma * 2.0, 10.0)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
