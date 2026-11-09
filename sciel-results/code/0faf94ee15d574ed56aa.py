# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with a deadline guard and a final (1+1)-ES polish pass."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        t0 = time.time()
        DEADLINE = 280.0  # safety margin before 300 s wall-clock kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # Initial point: center of the box (robust for BBOB-style shifted optima)
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # CMA-ES parameters
        # lambda: population size ~ 4 + floor(3 log n)
        lam = max(4, 4 + int(3 * np.log(dim)))
        # initial step size: ~3 * box half-range is too big; use 2.0
        sigma = 2.0
        # weights
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, lam + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w[:mu] ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (2 - c_c - c_sigma)
        c_mu = (2 / ((dim + 2) ** 2 + mu_eff)) * (
            min(1, c_1 / 2) - c_1
        )
        c_mu = max(c_mu, 0.0)

        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # we'll maintain via eigendecomp
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Precompute Cholesky-like: we use eigendecomp each step (dim=5, cheap)
        def update_chol():
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            return eigvecs, eigvals

        # initial
        eigvecs, eigvals = update_chol()

        for it in range(budget - calls):
            if time.time() - t0 > DEADLINE:
                break

            # sample offspring
            offspring = []
            for _ in range(lam):
                z = rng.standard_normal(dim)
                # x + sigma * C^{1/2} z  -> eigvecs * (sqrt(eigvals) * (eigvecs^T z))
                proj = eigvecs.T @ z
                step = eigvecs @ (np.sqrt(eigvals) * proj)
                y = x + sigma * step
                offspring.append(y)

            fvals = np.empty(lam)
            for i in range(lam):
                fvals[i] = func(offspring[i])
                calls += 1

            # sort
            order = np.argsort(fvals)
            y_sorted = [offspring[i] for i in order]
            f_sorted = fvals[order]

            # best
            if f_sorted[0] < fx:
                x_best = y_sorted[0]
                fx = f_sorted[0]

            # mean of top mu
            m_new = sum(w[i] * y_sorted[i] for i in range(mu))

            # update evolution paths
            # C^{1/2} (m_new - x) / sigma
            diff = (m_new - x) / sigma
            proj = eigvecs.T @ diff
            step_vec = eigvecs @ (np.sqrt(eigvals) * proj)

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step_vec
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * step_vec

            # update C
            h_sigma = (
                np.linalg.norm(p_sigma)
                / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))
                / chi_n
                < (1.4 + 2 / (dim + 1))
            )
            C = (1 - c_c * c_c * (2 - c_c) + c_1 + c_mu) * C
            C += c_c * (p_c * p_c - C)
            # rank-1
            C += c_1 * (step_vec * step_vec)
            # rank-mu
            for i in range(mu):
                d_i = (y_sorted[i] - x) / sigma
                proj_i = eigvecs.T @ d_i
                step_i = eigvecs @ (np.sqrt(eigvals) * proj_i)
                C += c_mu * w[i] * (step_i * step_i)

            # enforce symmetry and PSD
            C = 0.5 * (C + C.T)
            # numerical safeguard
            eigvals2, eigvecs2 = np.linalg.eigh(C)
            if np.min(eigvals2) < 1e-12:
                C = C + 1e-12 * np.eye(dim)
            eigvecs, eigvals = update_chol()

            # update sigma
            if h_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            # clamp sigma
            sigma = float(np.clip(sigma, 1e-9, 10.0))

            x = m_new

        # ---- Final polish pass: (1+1)-ES with tiny steps ----
        # Use remaining budget and time
        remaining = budget - calls
        if remaining > 0:
            sigma_p = max(1e-4, sigma * 0.05)
            x_p = x.copy()
            fx_p = fx
            for _ in range(remaining):
                if time.time() - t0 > DEADLINE:
                    break
                y = x_p + sigma_p * rng.standard_normal(dim)
                fy = func(y)
                calls += 1
                if fy < fx_p:
                    x_p, fx_p = y, fy
                    sigma_p *= 1.3
                else:
                    sigma_p *= 0.8
                    sigma_p = max(sigma_p, 1e-8)
            x, fx = x_p, fx_p

        return x, fx
# EVOLVE-BLOCK-END
