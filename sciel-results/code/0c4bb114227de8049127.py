"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        best_x = x.copy()
        best_f = func(x)
        calls_made = 1

        for it in range(budget - 1):
            # Generate population
            ys = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.randn(dim)
                ys[i] = x + sigma * (z @ C_sqrt.T)
                ys[i] = np.clip(ys[i], -5, 5)

            # Evaluate
            fs = np.array([func(y) for y in ys])
            calls_made += lambda_

            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()

            # Weighted mean
            y_w = np.sum(weights[:, None] * ys[:mu], axis=0)

            # Update step-size control
            h_sigma = (1 - (1 - c_sigma) ** (2 * (it + 1))) / (1 - (1 - c_sigma) ** (2 * (it + 1)))
            h_sigma = 1.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ y_w)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = min(sigma, 1e6)
            sigma = max(sigma, 1e-10)

            # Update covariance matrix
            h_c = (1 - (1 - c_c) ** (2 * (it + 1))) / (1 - (1 - c_c) ** (2 * (it + 1)))
            h_c = 1.0
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Rank-one and rank-mu updates
            C = (1 + c_1 * (2 - c_1 - c_mu) - c_1) * C
            C += c_1 * np.outer(p_c, p_c)
            C += c_mu * np.sum(weights[:, None] * np.outer(ys[:mu] - y_w, ys[:mu] - y_w), axis=0)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update C_sqrt = C^(1/2)
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-10, None)
            C_sqrt = evecs @ (np.sqrt(evals)[:, None] * evecs.T)

            # Update mean
            x = x + y_w
            x = np.clip(x, -5, 5)

            if calls_made >= budget:
                break

        return best_x, best_f


# EVOLVE-BLOCK-END
