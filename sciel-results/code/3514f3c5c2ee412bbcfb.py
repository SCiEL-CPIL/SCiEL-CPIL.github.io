# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for BBOB f16 (Weierstrass) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with given budget, respecting deadline."""
        n = self.dim
        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = max(4, int(4 + 3 * np.log(n)))
        # Weights: log-linear
        w = np.log(lam / 2) - np.arange(lam)
        w /= w.sum()
        mu = int(lam / 2)
        w = w[:mu]
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1,
                   2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_eigvals = np.ones(n)
        C_eigvecs = np.eye(n)
        d_cor = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_c
        sigma = sigma0
        x = x0.copy()
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        # Precompute Cholesky of C (initially identity)
        C_sqrt = np.eye(n)

        for gen in range(budget // lam + 1):
            if time.time() > deadline:
                break
            # Generate population
            Y = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * (C_eigvecs @ (C_eigvals ** 0.5) @ z)
                y = np.clip(y, -5, 5)
                Y.append(y)

            # Evaluate
            F = []
            for y in Y:
                if time.time() > deadline:
                    break
                fy = func(y)
                F.append(fy)
                calls += 1
            if len(F) < lam:
                break

            # Sort
            idx = np.argsort(F)
            Y_sorted = np.array([Y[i] for i in idx])
            F_sorted = np.array([F[i] for i in idx])

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()

            # Weighted mean
            m = Y_sorted[:mu] @ w[:, None]

            # Update sigma
            d_mean = (m - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_mean
            sigma *= np.exp(c_sigma / chi_n *
                            (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 50.0)

            # Update C (only if step is significant)
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (
                    2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                p_c = (1 - c_c) * p_c + np.sqrt(
                    c_c * (2 - c_c) * mu_eff) * (m - x) / sigma

            # New mean
            x_new = m

            # Update covariance
            delta_d = (1 + c_c) * np.dot(p_c, p_c) / n - 1
            C = (1 + c_1 * (n + 1) + c_mu * delta_d - c_1 - c_mu) * C \
                + c_1 * (n + 1 - c_1) * np.outer(p_c, p_c) \
                + c_mu * np.sum(w[:, None, None] *
                                np.einsum('i,ij,ik->ijk',
                                          Y_sorted[:mu] - x,
                                          Y_sorted[:mu] - x,
                                          Y_sorted[:mu] - x), axis=0) / n

            # Eigendecomposition (periodic for speed)
            if gen % 5 == 0:
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                C_eigvals = np.clip(C_eigvals, 1e-12, 1e12)
                # Ensure symmetry
                C = 0.5 * (C + C.T)

            x = x_new

            if calls >= budget:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin (limit is 330s)
        n = self.dim
        total_budget = self.budget

        # Multi-restart CMA-ES strategy
        # Phase 1: quick exploration with small population (fast initial progress)
        # Phase 2: medium CMA-ES
        # Phase 3: fine polishing

        best_x = None
        best_f = np.inf
        calls_used = 0
        rng = np.random.default_rng(42)

        # Restart schedule: (num_restarts, sigma0, budget_fraction)
        # We allocate budget to give fast early progress
        # Phase 1: 1 restart, small sigma, ~40% budget
        # Phase 2: 2 restarts, medium sigma, ~35% budget
        # Phase 3: 2 restarts, small sigma, ~25% budget

        phases = [
            # (num_restarts, sigma0, budget_frac)
            (1, 3.0, 0.40),
            (2, 2.0, 0.35),
            (2, 0.5, 0.25),
        ]

        for num_restarts, sigma0, frac in phases:
            phase_budget = int(total_budget * frac) - calls_used
            if phase_budget <= 0:
                break

            # Split among restarts
            per_restart = max(1, phase_budget // num_restarts)

            for r in range(num_restarts):
                if time.time() > deadline:
                    break
                if calls_used >= total_budget:
                    break

                # Choose starting point
                if best_x is not None and r > 0:
                    # Perturb around best
                    x0 = best_x + sigma0 * 0.3 * rng.standard_normal(n)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-4, 4, n)

                remaining = min(per_restart, total_budget - calls_used)
                if remaining < 10:
                    break

                bx, bf, c = self._cmaes(func, x0, sigma0, remaining, rng, deadline)
                calls_used += c

                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

                if time.time() > deadline:
                    break

        # Final check: make sure we haven't exceeded budget
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
