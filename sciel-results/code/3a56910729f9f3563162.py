"""Hierarchical CMA-ES with random restarts and refinement for Rastrigin rotated."""
import numpy as np
import time
import math


class Optimizer:
    """Hierarchical budget allocation CMA-ES for Rastrigin rotated in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _run_cmaes(self, func, x0, sigma0, max_calls, deadline, rng, dim):
        """Run CMA-ES starting from x0 with initial sigma, using at most max_calls evaluations.
        Returns (best_x, best_f, calls_used)."""
        best_x = x0.copy()
        best_f = func(best_x)
        calls_used = 1

        if max_calls <= 1:
            return best_x, best_f, calls_used

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1.0 / (21 * dim ** 2))

        mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        while calls_used < max_calls:
            if time.time() > deadline:
                break

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

            # Evaluate population
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= max_calls or time.time() > deadline:
                    break
                fvals[i] = func(pop[i])
                calls_used += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            if calls_used >= max_calls:
                break

            # Sort and select
            order = np.argsort(fvals)
            y = np.zeros((mu, dim))
            for i in range(mu):
                y[i] = pop[order[i]]

            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * y, axis=0)

            # Update paths
            v = (new_mean - mean) / sigma

            # Compute C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            ps = (1 - c_sigma) * ps + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ v)

            norm_ps = np.linalg.norm(ps)
            t_gen = calls_used / lam
            h_sigma = (norm_ps / math.sqrt(1 - (1 - c_sigma) ** (2 * t_gen))) / chi_n < 1.4 + 2 / (dim + 1)

            pc = (1 - c_c) * pc + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * v

            # Update covariance
            delta_h = (2 - c_c) * h_sigma - 1
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(pc, pc)
            for i in range(mu):
                zi = (y[i] - mean) / sigma
                C += c_mu * weights[i] * np.outer(zi, zi)

            C = (C + C.T) / 2

            # Update sigma
            sigma *= math.exp((c_sigma / (1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma)) * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            mean = new_mean

        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Broad random search to find good starting points
        n_init = 20
        candidates = []
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x.copy()))

        calls_used = n_init
        candidates.sort(key=lambda t: t[0])

        global_best_x = candidates[0][1].copy()
        global_best_f = candidates[0][0]

        # Phase 2: Hierarchical budget allocation
        # Allocate remaining budget among top candidates
        remaining = budget - calls_used

        # Use top 3 candidates with decreasing budget
        n_refine = min(3, len(candidates))
        # Budget fractions: 1/2, 1/4, 1/8 of remaining
        fractions = [0.5, 0.25, 0.125]

        for idx in range(n_refine):
            if calls_used >= budget or time.time() > deadline:
                break

            fx, x0 = candidates[idx]
            # Use a sigma appropriate for the basin
            sigma0 = 1.5 if idx == 0 else 1.0 if idx == 1 else 0.5

            alloc = int(remaining * fractions[idx])
            if alloc < 20:
                alloc = 20

            bx, bf, cu = self._run_cmaes(func, x0, sigma0, alloc, deadline, rng, dim)
            calls_used += cu

            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()

        # Phase 3: Final refinement on the best solution found
        remaining = budget - calls_used
        if remaining > 10 and time.time() < deadline:
            bx, bf, cu = self._run_cmaes(func, global_best_x, 0.1, remaining, deadline, rng, dim)
            calls_used += cu
            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()

        return (global_best_x, global_best_f)
