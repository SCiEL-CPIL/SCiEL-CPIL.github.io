"""CMA-ES optimizer for rotated Rastrigin f15 with hierarchical budget allocation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES with hierarchical budget allocation and refinement for BBOB f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf

        def _remaining():
            return budget - calls

        def _time_ok():
            return time.time() < deadline

        # Phase 1: Broad exploration with multiple random starts
        # Use a mix of ranges to cover the box
        n_starts = 15
        starts = []
        for i in range(n_starts):
            if not _time_ok() or calls >= budget:
                break
            # Mix of ranges: some wide, some medium
            if i < 5:
                x = rng.uniform(-4.5, 4.5, dim)
            elif i < 10:
                x = rng.uniform(-3.0, 3.0, dim)
            else:
                x = rng.uniform(-1.5, 1.5, dim)
            fx = func(x)
            calls += 1
            starts.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: Hierarchical CMA-ES on top candidates
        # Sort starts by fitness
        starts.sort(key=lambda t: t[0])
        
        # Allocate budget: top 3 get 1/4, 1/8, 1/16 of remaining
        remaining = _remaining()
        if remaining > 20:
            budgets = [remaining // 4, remaining // 8, remaining // 16]
            # Ensure minimum budget
            budgets = [max(b, 10) for b in budgets]
        else:
            budgets = [remaining] * min(3, len(starts))

        for k, (fx, x_start) in enumerate(starts[:3]):
            if not _time_ok() or calls >= budget:
                break
            b = budgets[k] if k < len(budgets) else 10
            if b < 10:
                continue
            # Run CMA-ES from this start
            best_x_k, best_f_k = self._cma_es_run(
                func, x_start, b, dim, rng, calls, deadline
            )
            calls += b
            if best_f_k < best_f:
                best_f = best_f_k
                best_x = best_x_k

        # Phase 3: Final refinement with small sigma
        if _time_ok() and calls < budget:
            remaining = _remaining()
            if remaining > 5:
                # Small sigma refinement around best
                sigma_refine = 0.1
                n_refine = min(remaining, 30)
                for _ in range(n_refine):
                    if not _time_ok() or calls >= budget:
                        break
                    x_try = best_x + sigma_refine * rng.standard_normal(dim)
                    x_try = np.clip(x_try, -5, 5)
                    fx_try = func(x_try)
                    calls += 1
                    if fx_try < best_f:
                        best_f = fx_try
                        best_x = x_try.copy()

        return best_x, best_f

    def _cma_es_run(self, func, x0, budget, dim, rng, calls_in, deadline):
        """Run CMA-ES from starting point x0 with given budget."""
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        mean = x0.copy()
        sigma = 0.8
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        best_x = x0.copy()
        best_f = func(x0)
        calls = 1

        while calls < budget and time.time() < deadline:
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except:
                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C)
                except:
                    break

            pop = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                pop[i] = mean + sigma * (L @ z)

            fvals = np.zeros(lam)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    fvals[i] = np.inf
                    continue
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            if calls >= budget:
                break

            # Sort
            order = np.argsort(fvals)
            y = np.zeros((mu, dim))
            for i in range(mu):
                y[i] = pop[order[i]]

            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * y, axis=0)

            # Update paths
            v = (new_mean - mean) / sigma if sigma > 1e-10 else np.zeros(dim)

            # Compute C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            ps = (1 - c_sigma) * ps + c_sigma / math.sqrt(c_sigma) * C_inv_sqrt @ v

            norm_ps = np.linalg.norm(ps)
            t_gen = calls / lam
            h_sigma = (norm_ps / math.sqrt(1 - (1 - c_sigma) ** (2 * t_gen))) / chi_n < 1.4 + 2 / (dim + 1)

            pc = (1 - c_c) * pc + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * v

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            delta = (1 + delta_h) * (1 - c_1)

            rank1 = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_norm = y[i] - mean
                rank_mu += weights[i] * np.outer(y_norm, y_norm)

            C = delta * C + c_1 * (rank1 - C) + c_mu * rank_mu / (sigma ** 2)
            C = (C + C.T) / 2

            # Update sigma
            sigma = sigma * math.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 0.01, 5.0)

            mean = new_mean

            # Early termination
            if sigma < 0.05 and calls > 10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
