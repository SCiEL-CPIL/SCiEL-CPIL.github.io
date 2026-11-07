import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    @staticmethod
    def _sobol(n, dim, rng):
        """Generate n points in [-5,5]^dim using a Halton-like low-discrepancy sequence."""
        # Use a simple Halton sequence (low-discrepancy) for stratified coverage
        pts = np.zeros((n, dim))
        # First few primes for Halton bases
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
        for d in range(dim):
            base = bases[d]
            for i in range(n):
                f = 1.0 / base
                val = 0.0
                ii = i + 1
                while ii > 0:
                    f /= base
                    val += f * (ii % base)
                    ii //= base
                pts[i, d] = -5.0 + 10.0 * val
        return pts

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s wall-clock limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return time.time() < deadline

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        best_x = None
        best_f = np.inf

        # ---------- Phase 1: Dense stratified exploration ----------
        # Büche-Rastrigin is multi-modal; dense early sampling gives fast AOCC progress.
        n_grid = 70
        grid_pts = self._sobol(n_grid, dim, rng)
        # Add a few deterministic anchors for robustness
        anchors = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
        ]
        candidates = list(grid_pts) + anchors

        for p in candidates:
            if calls >= budget or not time_ok():
                break
            p = clip(p)
            f = float(func(p))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = p.copy()

        if best_x is None:
            p = clip(rng.uniform(-5, 5, dim))
            best_f = float(func(p))
            calls += 1
            best_x = p.copy()

        # ---------- Phase 2: CMA-ES local refinement ----------
        if calls < budget and time_ok():
            best_x, best_f = self._cmaes(func, best_x, best_f, budget - calls, rng, deadline)

        return best_x, best_f

    def _cmaes(self, func, x0, f0, budget_left, rng, deadline):
        dim = self.dim
        lam = 6
        mu = lam // 2

        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        x_mean = x0.copy()
        sigma = 0.6
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        c_sigma = (mu + 0.5) / (dim + 3)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        calls_used = 0
        best_x = x0.copy()
        best_f = f0

        def time_ok():
            return time.time() < deadline

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def mat_sqrt_inv(M):
            # Inverse square root via eigen decomposition (symmetric positive definite)
            eigval, eigvec = np.linalg.eigh(M)
            eigval = np.clip(eigval, 1e-12, None)
            return eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T

        while calls_used < budget_left and time_ok():
            if calls_used + lam > budget_left:
                # Not enough budget for a full generation; do single-step perturbation
                while calls_used < budget_left and time_ok():
                    y = clip(x_mean + sigma * rng.standard_normal(dim))
                    fy = float(func(y))
                    calls_used += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        x_mean = y.copy()
                        sigma = min(sigma * 1.15, 5.0)
                    else:
                        sigma = max(sigma * 0.85, 1e-5)
                break

            # Generate offspring
            offspring = []
            for _ in range(lam):
                if calls_used >= budget_left or not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = clip(x_mean + sigma * (C_inv_sqrt @ z))
                fy = float(func(y))
                calls_used += 1
                offspring.append((y, fy))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])
            xs = np.array([o[0] for o in offspring[:mu]])
            fs = np.array([o[1] for o in offspring[:mu]])

            # Weighted recombination
            x_new = np.sum(w[:, None] * xs, axis=0)

            # Cumulation for sigma adaptation
            d_x = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_x

            # Cumulation for covariance adaptation
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * d_x

            # Hessian-free rank-mu + rank-1 update
            delta_sigma = np.linalg.norm(p_sigma) / chi_n - (1 - c_sigma)
            h_sigma = 1.0 if delta_sigma < (1 + 1.5 / c_sigma) * chi_n else 0.0

            C = (1 + c_1 * (h_sigma * (mu_eff - 1) - 1) + c_mu) * C \
                - c_1 * (1 - h_sigma) * np.outer(p_c, p_c) \
                - c_1 * mu_eff * np.eye(dim) \
                + c_mu * np.sum(w[:, None] * np.outer(xs - x_mean, xs - x_mean) / sigma ** 2, axis=0)

            # Symmetrize and ensure positive definiteness
            C = 0.5 * (C + C.T)
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.clip(eigval, 1e-10, None)
            C = eigvec @ np.diag(eigval) @ eigvec.T

            # Step-size adaptation
            sigma *= np.exp((c_sigma / 5.0) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-6, 10.0))

            # Update inverse square root
            C_inv_sqrt = mat_sqrt_inv(C)
            x_mean = x_new

        return best_x, best_f
