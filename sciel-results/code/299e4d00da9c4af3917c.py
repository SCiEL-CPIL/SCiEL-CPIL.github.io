import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        dim = self.dim
        budget = self.budget

        # CMA-ES configuration tuned for 5-dim bounded optimization
        lam = 12
        mu = lam // 2

        w = np.log(np.arange(1, mu + 1) + 0.5) - np.log(mu + 0.5)
        w = w / np.sum(w)
        mu_eff = np.sum(w) ** 2 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = 3.0 / (dim + 4.0)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # State
        m = np.zeros(dim)
        sigma = 1.5
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x = m.copy()
        best_f = np.inf
        calls = 0

        def eval_point(x):
            nonlocal calls, best_f, best_x
            calls += 1
            f = func(np.clip(x, -5, 5))
            if f < best_f:
                best_f = f
                best_x = np.clip(x, -5, 5).copy()
            return f

        # Optional: cheap initial random evaluations to get a good starting point
        # We'll let CMA-ES handle it, but start with a small random sample to find a reasonable mean
        # Actually, let's start CMA-ES directly but with a slightly larger sigma.

        while calls < budget:
            if time.time() >= deadline:
                break

            # 1. Sample population
            Z = np.random.standard_normal((lam, dim))

            # Cholesky decomposition is cheaper than eigh for updating
            # But we need C^(-1/2) for CSA. Let's use eigh once.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-14, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            Y = Z @ C_sqrt.T
            X = m + sigma * Y

            # Evaluate population
            F = np.array([eval_point(x) for x in X])

            # 2. Sort
            idx_sort = np.argsort(F)
            X_sorted = X[idx_sort]
            F_sorted = F[idx_sort]

            # 3. Update Mean
            m_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)

            # 4. Update Step Size (CSA)
            c_1_mu = c_1 + c_mu
            delta = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(c_1_mu * (2 - c_1_mu)) * (C_inv_sqrt @ delta)

            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (25 * dim ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))

            # 5. Update Covariance
            p_c = (1 - c_c) * p_c + c_c * delta

            Y_sorted = (X_sorted[:mu] - m) / sigma
            C_new = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C_new += c_mu * w[i] * np.outer(Y_sorted[i], Y_sorted[i])

            C_new = (C_new + C_new.T) / 2

            m = m_new
            C = C_new

        return best_x, best_f
