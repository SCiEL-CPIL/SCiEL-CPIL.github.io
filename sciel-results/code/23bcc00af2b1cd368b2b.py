import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        n = self.dim
        # Standard CMA-ES parameters for low dimension
        lam = 4 + int(3 * np.log(n))
        if lam < 2:
            lam = 2
        mu = lam // 2

        # Weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)

        # Strategy parameters
        c_sigma = (2 + np.sqrt(2 * lam)) / (lam + 2)
        c_c = (4 + lam / mu - 2) / (lam + 5)
        c_1 = 2 / (lam ** 2 + 18)
        c_mu_raw = np.sum(weights) / (lam - np.sum(weights))
        c_mu = c_mu_raw * (2 + c_mu_raw) / (2 + c_1)
        d_sigma = 1 + 2 * max(0, (c_mu - 1) / (lam - 2)) + c_mu

        # Initial state
        x_mean = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        C_sqrt = np.eye(n)

        best_x = x_mean.copy()
        best_f = func(best_x)
        evals = 1

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        while evals + lam <= self.budget:
            if time.time() > deadline:
                break

            # Generate population
            zs = np.random.standard_normal((lam, n))
            ys = zs @ C_sqrt.T
            xs = x_mean[None, :] + sigma * ys
            fs = np.array([func(x) for x in xs])
            evals += lam

            # Update best
            min_idx = np.argmin(fs)
            if fs[min_idx] < best_f:
                best_f = fs[min_idx]
                best_x = xs[min_idx].copy()

            # Sort
            order = np.argsort(fs)
            y_sorted = ys[order]
            x_sorted = xs[order]

            # Weighted mean
            y_w = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            x_w = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

            # Update mean
            x_mean = x_w

            # Update path p_c
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(c_mu / (2 - c_mu)) * y_w

            # Update covariance matrix
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c) ** 2) * C)
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(y_sorted[i], y_sorted[i])

            # Symmetrize
            C = (C + C.T) / 2

            # Update C_sqrt via eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # Update sigma
            t = evals // lam
            chi_n = np.sqrt(1 - (1 - c_sigma) ** (2 * t))
            
            # h_sigma check
            h_sigma = 1 if np.linalg.norm(p_sigma) < (1.4 + 2 / (lam + 1)) * chi_n else 0

            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_mu) * y_w

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

        return best_x, best_f
