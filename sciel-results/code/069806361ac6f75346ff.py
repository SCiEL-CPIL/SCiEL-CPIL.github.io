"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f18 (Schaffers F7 ill-conditioned)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # CMA-ES parameters
        lambda_ = 4 * int(np.ceil(3 + np.log(dim)))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (4 * (dim - 1))) / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = np.clip(1 - c_1 - 2 * mu_eff / ((dim + 2) ** 2 - 1), 0, (2 - c_1) / 2)

        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        best_f = evaluate(x_mean)

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            Z = rng.standard_normal((lambda_, dim))
            Y = x_mean[None, :] + sigma * (Z @ np.diag(D) @ B.T)

            # Evaluate
            F = np.array([evaluate(y) for y in Y])

            if calls >= budget or time.time() > deadline:
                break

            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Weighted mean of best mu
            y_w = np.sum(w[:, None] * Y[:mu], axis=0)

            # Step size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (D @ (B.T @ ((y_w - x_mean) / sigma)))
            hsig = np.sqrt((1 - np.exp(-2 * calls / (dim * c_sigma))) / (1 - np.exp(-2 / (dim * c_sigma))))
            if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * hsig:
                c_sigma_eff = c_sigma * (1 - np.exp(-calls / (dim * c_sigma)))
                x_mean = x_mean + c_sigma_eff * sigma * (B @ (D * ps))

            # Covariance matrix update
            diff = (y_w - x_mean) / sigma
            diff = B @ (D * diff)
            C = (1 + c_1 * c_mu) * C - c_1 * np.outer(diff, diff)

            for i in range(mu):
                diff_i = (Y[i] - x_mean) / sigma
                diff_i = B @ (D * diff_i)
                C += c_mu * w[i] * np.outer(diff_i, diff_i)

            # Periodic eigen-decomposition to maintain PSD and update B, D
            if calls % 10 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)

            # Step size adaptation
            sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(ps) / chi_n - 1) / 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
