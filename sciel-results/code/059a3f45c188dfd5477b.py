"""Optimizer for BBOB f13 (Sharp ridge) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


class Optimizer:
    """CMA-ES black-box optimizer with time guard and warm start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        deadline = time.time() + 250.0  # safety margin below 300s

        # ---------- Phase 0: warm start (front-load early progress) ----------
        # A short coarse random search to get a reasonable starting point.
        # We spend a small fixed number of calls here.
        warm_calls = min(30, max(10, budget // 20))
        best_x = np.zeros(dim)
        best_f = np.inf
        for _ in range(warm_calls):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # ---------- Phase 1: CMA-ES ----------
        x = best_x.copy()
        f_best = best_f
        calls = warm_calls

        # Initial step size: moderate, will adapt.
        sigma = 1.0

        # CMA-ES parameters
        pop = 4 + int(2 * math.sqrt(dim))  # population size
        pop = max(4, pop)

        # Means for adaptation
        mu = pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 * min(mu_eff / 3, 1) / dim) / (dim ** 1.6 + mu_eff + 13)
        c_mu = min(1 - c_1,
                  2 * ((mu_eff + 2) / (dim + 2) - 1) / (dim ** 1.6 + mu_eff + 13))

        C = np.eye(dim)
        # Precompute sqrt(C) for sampling (start as identity)
        sqrtC = np.eye(dim)
        sqrtC_inv = np.eye(dim)

        # Accumulators for covariance update
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        def sample_sigma():
            return sigma * sqrtC @ rng.standard_normal(dim)

        def update_sqrtC():
            nonlocal sqrtC, sqrtC_inv
            # Cholesky-ish update of C, then recompute sqrt via eigendecomp
            # (dim is small, so eigendecomposition is fine and stable)
            w = np.linalg.eigvalsh(C)
            w = np.clip(w, 1e-12, None)
            # Reconstruct C from eigenvalues (C is symmetric PSD)
            # Actually we need the eigenvectors too. Use full eig.
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-12, None)
            sqrtC = evecs @ (np.sqrt(evals)[:, None] * evecs.T)
            sqrtC_inv = evecs @ (1.0 / np.sqrt(evals)[:, None] * evecs.T)
            return sqrtC, sqrtC_inv

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            ys = np.empty((pop, dim))
            for i in range(pop):
                y = x + sample_sigma()
                ys[i] = np.clip(y, -5, 5)

            # Evaluate
            fs = np.empty(pop)
            for i in range(pop):
                if time.time() > deadline:
                    break
                fs[i] = func(ys[i])
                calls += 1
                if fs[i] < f_best:
                    f_best = fs[i]
                    best_x = ys[i].copy()
            else:
                pass

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # New mean: weighted average of best mu
            x_new = (weights[:, None] * ys_sorted[:mu]).sum(axis=0)

            # Update path for sigma
            y_sigma = sqrtC_inv @ (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            chi_n = math.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (24 * dim ** 2))
            if np.linalg.norm(ps) / math.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls))) > (1.4 + 1 / dim) * chi_n:
                pass  # don't reduce sigma based on this check in standard CMA; skip

            sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update path for C
            y_c = sqrtC_inv @ (x_new - x)
            pc = (1 - c_c) * pc + math.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

            # Covariance update
            H = (pc[:, None] * pc[None, :]) / c_c - C
            delta_h = (1 + c_c - c_1) * (2 - c_c) * mu_eff
            H += (c_1 / c_c) * (delta_h - 1 + (c_1 / c_c) * (mu_eff - 1)) * C
            # Rank-mu update
            H += c_mu * sum(
                (weights[i] * (ys_sorted[i] - x)[:, None] * (ys_sorted[i] - x)[None, :])
                for i in range(mu)
            )
            C = (1 + c_1 * (delta_h - 1 + c_1) + c_mu) * C - c_1 * H
            # Symmetrize
            C = 0.5 * (C + C.T)

            x = x_new

            # Recompute sqrt(C) periodically (every iteration is fine for dim=5)
            sqrtC, sqrtC_inv = update_sqrtC()

            if calls >= budget:
                break

        return best_x, f_best
# EVOLVE-BLOCK-END
