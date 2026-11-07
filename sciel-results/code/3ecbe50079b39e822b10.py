"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (full covariance) with deadline guard and simple restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_cmaes(self, func, x0, f0, sigma0, budget, deadline, rng):
        dim = self.dim
        x = x0.copy()
        f = f0
        sigma = sigma0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Standard CMA-ES parameters
        mu = max(1, dim // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (2 + 2.0 / (dim + 1)) / (dim + 4 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 9 + 2 * mu_eff / dim)
        d_amp = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        def _cholesky_update():
            nonlocal C, C_inv, C_sqrt, C_sqrt_inv
            C = np.atleast_2d(C)
            C = 0.5 * (C + C.T)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Symmetrize and add jitter
                C = C + 1e-12 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: eigen-based
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.clip(vals, 1e-12, None)
                    C = (vecs * vals) @ vecs.T
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            C_sqrt = L
            C_sqrt_inv = np.linalg.inv(L)
            C_inv = C_sqrt_inv.T @ C_sqrt_inv
            C_inv = 0.5 * (C_inv + C_inv.T)

        _cholesky_update()

        calls = 1  # initial
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Sample candidates
            ys = []
            fs = []
            zs = []
            for _ in range(mu):
                z = rng.standard_normal(dim)
                zs.append(z)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                fs.append(fy)

            ys = np.array(ys)
            fs = np.array(fs)
            zs = np.array(zs)

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            zs = zs[order]

            # Update best
            if fs[0] < f:
                f = fs[0]
                x = ys[0].copy()

            # Weighted mean
            y_w = np.sum(w[:, None] * ys, axis=0)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (y_w - x))
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - x)

            x = y_w

            # Rank-mu and rank-1 updates
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                zi = C_sqrt_inv @ (ys[i] - y_w)
                rank_mu += w[i] * np.outer(zi, zi)

            C = (1 + c_sigma * delta_h + c_sigma * mu_eff - 1) * C \
                + c_sigma * np.outer(ps, ps) \
                + c_c * delta_h * np.outer(pc, pc) \
                + (1 - c_sigma * delta_h - c_sigma * mu_eff) * rank_mu

            _cholesky_update()

            # Adapt sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

        return x, f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s / 330s kill

        rng = np.random.default_rng(0)
        dim = self.dim

        # Multi-start: first a global-ish CMA-ES, then local refinements
        # Initial point near 0 (Rosenbrock minimum is near (1,1,...,1) pre-rotation)
        x0 = rng.normal(0, 0.5, dim)
        f0 = func(x0)
        calls_used = 1

        best_x = x0.copy()
        best_f = f0

        # First phase: larger sigma to find the valley (global)
        # Budget split: use most of the budget on one strong CMA-ES run,
        # with a couple of restarts if it stalls.
        sigma0 = 1.5
        budget = self.budget

        # Give the first run a generous share
        phase_budget = int(budget * 0.6)
        x, f, c = self._run_cmaes(func, x0, f0, sigma0, phase_budget, deadline, rng)
        calls_used += c
        if f < best_f:
            best_f = f
            best_x = x.copy()

        if time.time() > deadline or calls_used >= budget:
            return best_x, best_f

        # Second phase: local refinement around the best with small sigma
        remaining = budget - calls_used
        sigma_local = 0.1
        x2, f2, c2 = self._run_cmaes(func, best_x.copy(), best_f, sigma_local, remaining, deadline, rng)
        calls_used += c2
        if f2 < best_f:
            best_f = f2
            best_x = x2.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
