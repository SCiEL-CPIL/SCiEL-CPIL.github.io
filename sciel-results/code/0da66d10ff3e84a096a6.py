"""Optimizer for BBOB-style f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, start_x, sigma_init, max_evals, deadline, global_best, calls_ref):
        """Run one CMA-ES instance. Returns (best_x, best_fx, n_calls)."""
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))  # ~13 for dim=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        x = np.array(start_x, dtype=float, copy=True)
        sigma = sigma_init
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x = x.copy()
        best_fx = func(x)
        n_calls = 1
        calls_ref[0] += 1
        global_best[0] = min(global_best[0], best_fx)

        gen = 0
        no_improve = 0
        threshold = 1.5 * np.sqrt(mu_eff)

        while n_calls < max_evals and time.time() < deadline:
            gen += 1

            # Sample population
            Z = np.random.randn(lam, dim)
            # Transform to covariance structure
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_eig = np.sqrt(eigvals)
                # Y = x + sigma * eigvecs * (sqrt_eig * (eigvecs.T @ Z.T)).T
                transformed = (eigvecs * sqrt_eig) @ (eigvecs.T @ Z.T).T
                Y = x[None, :] + sigma * transformed
            except np.linalg.LinAlgError:
                Y = x[None, :] + sigma * Z

            Y = np.clip(Y, -5, 5)

            # Evaluate population
            F = np.empty(lam)
            for i in range(lam):
                if n_calls >= max_evals or time.time() >= deadline:
                    break
                F[i] = func(Y[i])
                n_calls += 1
                calls_ref[0] += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = Y[i].copy()
                    global_best[0] = min(global_best[0], best_fx)
                    no_improve = 0
                else:
                    no_improve += 1

            n_eval = np.sum(np.isfinite(F))
            if n_eval < 2:
                break

            # Sort
            order = np.argsort(F[:n_eval])
            Y_sorted = Y[:n_eval][order]
            F_sorted = F[:n_eval][order]

            # Update mean
            x_new = sum(weights[i] * Y_sorted[i] for i in range(mu))
            x_new = np.clip(x_new, -5, 5)

            # Update evolution paths
            ydiff = x_new - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (ydiff / sigma)

            # Hessian sum
            H = sum(weights[i] * np.outer((Y_sorted[i] - x) / sigma, (Y_sorted[i] - x) / sigma)
                    for i in range(mu))

            # h_sigma (simplified)
            h_sigma = 1.0
            if gen > 5:
                if np.linalg.norm(p_sigma) / np.sqrt(mu_eff) > 1.4 * np.sqrt(2 * dim + 1):
                    h_sigma = 0.0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (ydiff / sigma)

            # Update C
            C = (1 + c_1 + c_mu - c_1 - mu_eff * c_mu) * C + c_1 * (np.outer(p_c, p_c) - C) + c_mu * H
            C = (C + C.T) / 2

            # Ensure PSD
            eigvals2, eigvecs2 = np.linalg.eigh(C)
            eigvals2 = np.maximum(eigvals2, 1e-12)
            C = eigvecs2 @ np.diag(eigvals2) @ eigvecs2.T

            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x = x_new

            # Stagnation check
            if no_improve > 3 * dim and sigma < 0.005:
                break

            # Budget check
            if n_calls >= max_evals:
                break

        return best_x, best_fx, n_calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        total_budget = self.budget
        global_best = [np.inf]
        calls_ref = [0]

        # Initial random sampling for early progress
        n_init = min(15, total_budget // 4)
        best_x = np.zeros(dim)
        best_fx = func(best_x)
        calls_ref[0] += 1
        global_best[0] = best_fx

        rng = np.random.RandomState(123)
        for _ in range(n_init - 1):
            if calls_ref[0] >= total_budget or time.time() >= deadline:
                break
            x_try = rng.uniform(-4, 4, size=dim)
            f_try = func(x_try)
            calls_ref[0] += 1
            if f_try < best_fx:
                best_fx = f_try
                best_x = x_try.copy()
                global_best[0] = best_fx

        # CMA-ES with adaptive restarts
        # Use smaller lambda for faster early progress
        lam_cma = 4 + int(3 * np.log(dim))  # ~13
        remaining = total_budget - calls_ref[0]

        # Multiple restarts with decreasing sigma
        n_restarts = 4
        # Budget allocation: 50%, 25%, 15%, 10%
        allocs = [int(remaining * 0.50), int(remaining * 0.25), int(remaining * 0.15), int(remaining * 0.10)]
        # Adjust to fit
        total_alloc = sum(allocs)
        diff = remaining - total_alloc
        allocs[0] += diff

        for r in range(n_restarts):
            if calls_ref[0] >= total_budget or time.time() >= deadline:
                break

            rb = min(allocs[r], total_budget - calls_ref[0])
            if rb < lam_cma:
                break

            if r == 0:
                start_x = best_x.copy()
                sigma_init = 1.0
            else:
                # Perturb from current best
                pert = 0.5 / (r + 1) * np.random.randn(dim)
                start_x = np.clip(best_x + pert, -5, 5)
                sigma_init = 0.5 / (r + 1)

            bx, bfx, ncalls = self._cma_es(func, start_x, sigma_init, rb, deadline, global_best, calls_ref)
            if bfx < best_fx:
                best_fx = bfx
                best_x = bx.copy()

        # Final polish: use remaining for a tight local search
        if calls_ref[0] < total_budget and time.time() < deadline:
            polish_budget = total_budget - calls_ref[0]
            if polish_budget >= lam_cma:
                sigma_polish = max(0.01, 0.05)
                bx, bfx, ncalls = self._cma_es(func, best_x.copy(), sigma_polish, polish_budget,
                                               deadline, global_best, calls_ref)
                if bfx < best_fx:
                    best_fx = bfx
                    best_x = bx.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
