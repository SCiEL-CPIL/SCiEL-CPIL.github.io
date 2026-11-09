"""CMA-ES with Rastrigin-aware restarts for rotated f15."""
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES with periodic restarts exploiting Rastrigin periodicity."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cmaes_once(self, func, x0, sigma0, max_calls, deadline, rng, dim):
        """Single CMA-ES run. Returns (best_x, best_f, calls_used)."""
        best_x = x0.copy()
        best_f = func(best_x)
        calls_used = 1

        if max_calls <= 1:
            return best_x, best_f, calls_used

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
        n_stagnant = 0
        stagnation_threshold = 15  # generations without improvement

        while calls_used < max_calls:
            if time.time() > deadline:
                break

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
                if calls_used >= max_calls or time.time() > deadline:
                    break
                fvals[i] = func(pop[i])
                calls_used += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            if calls_used >= max_calls:
                break

            order = np.argsort(fvals)
            y = np.zeros((mu, dim))
            for i in range(mu):
                y[i] = pop[order[i]]

            new_mean = np.sum(weights[:, np.newaxis] * y, axis=0)
            v = (new_mean - mean) / sigma

            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            ps = (1 - c_sigma) * ps + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ v)
            norm_ps = np.linalg.norm(ps)
            t_gen = calls_used / lam
            h_sigma = (norm_ps / math.sqrt(1 - (1 - c_sigma) ** (2 * t_gen))) / chi_n < 1.4 + 2 / (dim + 1)
            pc = (1 - c_c) * pc + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * v

            delta_h = (2 - c_c) * h_sigma - 1
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(pc, pc)
            for i in range(mu):
                zi = (y[i] - mean) / sigma
                C += c_mu * weights[i] * np.outer(zi, zi)
            C = (C + C.T) / 2

            sigma *= math.exp((c_sigma / (1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma)) * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            mean = new_mean

            # Stagnation detection
            if fvals[order[0]] >= best_f:
                n_stagnant += 1
            else:
                n_stagnant = 0

            # If significantly stagnant, shrink sigma and nudge toward periodic minima
            if n_stagnant > stagnation_threshold:
                sigma = max(sigma * 0.5, 0.05)
                # Nudge mean toward nearest integer (Rastrigin minima are at integers)
                rounded = np.round(mean)
                mean = 0.7 * mean + 0.3 * rounded
                n_stagnant = 0

        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        global_best_x = None
        global_best_f = np.inf
        calls_used = 0

        # Phase 1: Initial sampling
        n_init = 30
        for i in range(n_init):
            if calls_used >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls_used += 1
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

        # Phase 2: Multiple CMA-ES runs with restarts
        # Use remaining budget for CMA-ES, restarting if stuck
        max_cmaes_calls = budget - calls_used
        n_restarts = 4
        calls_per_restart = max(50, max_cmaes_calls // n_restarts)

        for r in range(n_restarts):
            if calls_used >= budget or time.time() > deadline:
                break

            if r == 0:
                x0 = global_best_x.copy()
                sigma0 = 1.5
            else:
                # Restart near best but with perturbation
                if global_best_x is not None:
                    x0 = global_best_x + rng.normal(0, 0.8, dim)
                    x0 = np.clip(x0, -4.5, 4.5)
                else:
                    x0 = rng.uniform(-4.5, 4.5, dim)
                sigma0 = 1.0

            alloc = min(calls_per_restart, budget - calls_used)
            if alloc < 20:
                break

            bx, bf, cu = self._cmaes_once(func, x0, sigma0, alloc, deadline, rng, dim)
            calls_used += cu

            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()

        # Phase 3: Final fine-tuning
        remaining = budget - calls_used
        if remaining > 20 and global_best_x is not None and time.time() < deadline:
            bx, bf, cu = self._cmaes_once(func, global_best_x, 0.1, remaining, deadline, rng, dim)
            calls_used += cu
            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()

        return (global_best_x, global_best_f)
