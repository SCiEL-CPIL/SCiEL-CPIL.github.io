"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer: random spread + CMA-ES + quadratic polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        calls = 0

        def f(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            calls += 1
            val = func(x)
            return val

        # Phase 1: Quick random sampling to find a good starting point
        n_init = min(30, budget)
        x = np.zeros((n_init, dim))
        fx = np.zeros(n_init)
        for i in range(n_init):
            x[i] = rng.uniform(-5, 5, dim)
            fx[i] = f(x[i])

        best_idx = np.argmin(fx)
        x_best = x[best_idx].copy()
        f_best = fx[best_idx]

        # Phase 2: CMA-ES
        if calls < budget - 10:
            x_best, f_best = self._cmaes(func, x_best, f_best, calls, budget, deadline, rng)

        # Phase 3: Quadratic polish (Newton-style for Sphere)
        if calls < budget - 10 and time.time() < deadline:
            x_best, f_best = self._polish(func, x_best, f_best, calls, budget, deadline, rng)

        return x_best, f_best

    def _cmaes(self, func, x0, f0, calls, budget, deadline, rng):
        dim = self.dim
        x = x0.copy()
        f_best = f0
        sigma = 1.0

        # CMA-ES parameters
        N = min(4 * dim + 2, 8)  # population size
        C1 = (2 + 2 / (dim + 1.7)) / ((dim + 1)**2 + 2)
        Cmua = min(0.4, C1 * 2 / 3)
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = C1
        c_mu = Cmua
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        D = 1.0
        B = np.eye(dim)
        B_inv = np.eye(dim)
        eigenvalues = np.ones(dim)

        def update_C(sigma):
            nonlocal C, B, B_inv, D, eigenvalues
            # Covariance matrix update
            w = np.zeros((N, dim))
            for i in range(mu):
                w[i] = weights[i] * (pop[i] - x) / sigma
            C += -c_1 * C + c_mu * (w.T @ w) + (c_1 * (1 - c_1) * (1 + 2 * c_mu / c_1) * sigma**2 * (1 - np.trace(C) / dim)) * np.eye(dim)
            # Eigen decomposition
            eigenvalues, B = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            D = 1.0 / np.sqrt(eigenvalues)
            B_inv = B.T

        pop = np.zeros((N, dim))
        fpop = np.zeros(N)

        while calls < budget - 10 and time.time() < deadline:
            # Generate population
            for i in range(N):
                z = rng.standard_normal(dim)
                pop[i] = x + sigma * (B @ (D * z))
                pop[i] = np.clip(pop[i], -5, 5)
                fpop[i] = func(pop[i])
                calls += 1

            # Sort by fitness
            idx = np.argsort(fpop)
            pop = pop[idx]
            fpop = fpop[idx]

            # Update best
            if fpop[0] < f_best:
                f_best = fpop[0]
                x_best = pop[0].copy()

            # Update mean
            m_old = x.copy()
            x = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (B_inv @ (x - m_old))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B_inv @ (x - m_old))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 1e8)

            # Update covariance
            update_C(sigma)

        return x_best, f_best

    def _polish(self, func, x0, f0, calls, budget, deadline, rng):
        dim = self.dim
        x = x0.copy()
        f_best = f0

        # Estimate gradient and Hessian using finite differences
        h = 1e-4
        max_iter = min(20, budget - calls)

        for _ in range(max_iter):
            if time.time() > deadline or calls >= budget - 2:
                break

            # Compute gradient
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                xp = np.clip(xp, -5, 5)
                xm = np.clip(xm, -5, 5)
                fp = func(xp)
                fm = func(xm)
                calls += 2
                grad[i] = (fp - fm) / (2 * h)

            # Compute Hessian
            H = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(i, dim):
                    xpp = x.copy()
                    xpm = x.copy()
                    xmp = x.copy()
                    xmm = x.copy()
                    xpp[i] += h
                    xpp[j] += h
                    xpm[i] += h
                    xpm[j] -= h
                    xmp[i] -= h
                    xmp[j] += h
                    xmm[i] -= h
                    xmm[j] -= h
                    xpp = np.clip(xpp, -5, 5)
                    xpm = np.clip(xpm, -5, 5)
                    xmp = np.clip(xmp, -5, 5)
                    xmm = np.clip(xmm, -5, 5)
                    fpp = func(xpp)
                    fpm = func(xpm)
                    fmp = func(xmp)
                    fmm = func(xmm)
                    calls += 4
                    H[i, j] = (fpp - fpm - fmp + fmm) / (4 * h**2)
                    H[j, i] = H[i, j]

            # Newton step
            try:
                step = np.linalg.solve(H + 1e-6 * np.eye(dim), grad)
            except np.linalg.LinAlgError:
                break

            # Line search
            alpha = 1.0
            for _ in range(20):
                x_new = np.clip(x - alpha * step, -5, 5)
                f_new = func(x_new)
                calls += 1
                if f_new < f_best:
                    x = x_new
                    f_best = f_new
                    break
                alpha *= 0.5
            else:
                break

        return x, f_best
# EVOLVE-BLOCK-END
