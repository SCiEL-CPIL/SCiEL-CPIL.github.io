"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        N = self.dim
        budget = self.budget
        start_time = time.time()
        timeout = 250.0  # Safety margin for 300s limit

        # CMA-ES Parameters
        lambda_ = 4
        mu = 2
        w = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)

        c_sigma = (mueff + 2) / (N + mueff + 5)
        c_c = 4 / (N + 4)
        c_1 = 2 / ((N + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 / ((N + 2) ** 2 + mueff))

        m = np.zeros(N)
        sigma = 2.0
        C = np.eye(N)
        p_sigma = np.zeros(N)
        p_c = np.zeros(N)

        evals = 0
        best_x = None
        best_f = np.inf

        phi_N = np.sqrt(N) * (1 - 1.0 / (4.0 * N) - 1.0 / (25.0 * N ** 2))

        while evals < budget:
            if time.time() - start_time > timeout:
                break

            # Sample population
            L = np.linalg.cholesky(C)
            zs = L @ np.random.randn(lambda_, N).T
            xs = (m + sigma * zs).T
            xs = np.clip(xs, -5, 5)

            fs = np.array([func(x) for x in xs])
            evals += len(xs)

            # Update best
            idx_best = np.argmin(fs)
            if fs[idx_best] < best_f:
                best_f = fs[idx_best]
                best_x = xs[idx_best].copy()

            # Sort
            idx = np.argsort(fs)
            xs_sorted = xs[idx]
            fs_sorted = fs[idx]

            # Update Mean
            m_new = m + sigma * np.sum(w[:, np.newaxis] * (xs_sorted[:mu] - m), axis=0)

            # Update Paths
            y_w = (m_new - m) / sigma

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_w

            # h_sigma condition (simplified)
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if norm_p_sigma < (1 + 2.0 / (N + 1.0)) * phi_N else 0.0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_w

            # Update Covariance
            sum_yy = np.zeros((N, N))
            for i in range(mu):
                y_i = (xs_sorted[i] - m) / sigma
                sum_yy += w[i] * np.outer(y_i, y_i)

            delta_sig = c_sigma * (2 - c_sigma) * mueff * (1 - h_sigma)

            C_new = (1 - c_1 - c_mu) * C + \
                    c_1 * (np.outer(p_c, p_c) + (1 - delta_sig) * C) + \
                    c_mu * sum_yy

            C_new = (C_new + C_new.T) / 2.0

            # Update Step Size
            sigma = sigma * np.exp(c_sigma * (norm_p_sigma / phi_N - 1.0))

            m = m_new
            C = C_new

            # Check positive definiteness
            try:
                np.linalg.cholesky(C)
            except:
                C = C + 1e-10 * np.eye(N)

        return best_x, best_f
# EVOLVE-BLOCK-END
