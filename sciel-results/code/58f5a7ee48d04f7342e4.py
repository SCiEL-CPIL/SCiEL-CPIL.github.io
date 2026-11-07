"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with restarts and Nelder-Mead polish for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cmaes_run(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with sigma0, using up to budget calls."""
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        calls_used = 0
        best_x = x.copy()
        best_f = func(x)
        calls_used += 1

        stagnation_count = 0
        no_improve_threshold = 12

        for gen in range(2000):
            if calls_used >= budget or time.time() >= deadline:
                break

            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            pop = np.array([x + sigma * (z @ L) for z in rng.standard_normal((lam, dim))])
            pop = np.clip(pop, -5, 5)
            fits = np.array([func(p) for p in pop])
            calls_used += lam

            idx_min = np.argmin(fits)
            if fits[idx_min] < best_f:
                best_f = fits[idx_min]
                best_x = pop[idx_min].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            sorted_idx = np.argsort(fits)
            y_sorted = pop[sorted_idx]

            m_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            yw = (m_new - x) / sigma
            pc = (1 - cs) * pc + cs * yw
            ps = (1 - cs) * ps + cs * np.sign(yw)

            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi
            sigma *= np.exp(cs / chi * (np.linalg.norm(pc) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            delta_hsig = (1.0 if hsig else 0.0) * (1 - cs) * cs
            C = (1 + c1 + cmu - c1 - cmu) * C
            C += c1 * (np.outer(yw, yw) + (1.0 if hsig else 0.0) * cs * (1 - cs) - C)
            C += cmu * np.sum(weights[:, None] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x) / (sigma ** 2), axis=0)
            C = (C + C.T) / 2

            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals) + 1e-12) * np.eye(dim)

            x = m_new

            if stagnation_count >= no_improve_threshold and calls_used < budget - lam:
                break

        return best_x, best_f, calls_used

    def _nelder_mead_polish(self, func, x0, max_calls, deadline):
        """Polish with Nelder-Mead, tracking call count."""
        calls = [0]
        
        def wrapped_func(x):
            calls[0] += 1
            if calls[0] > max_calls or time.time() >= deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            return func(x)

        try:
            result = minimize(wrapped_func, x0, method='Nelder-Mead',
                            options={'maxiter': max_calls, 'xatol': 1e-10, 'fatol': 1e-12})
            if calls[0] <= max_calls and result.fun < np.inf:
                return result.x, result.fun, calls[0]
        except Exception:
            pass
        return x0, np.inf, calls[0]

    def __call__(self, func):
        deadline = time.time() + 325.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        calls_remaining = budget

        # Multi-restart CMA-ES
        restart_configs = [
            (np.zeros(dim), 2.0),
            (np.zeros(dim), 1.0),
            (rng.uniform(-3, 3, dim), 1.5),
            (rng.uniform(-4, 4, dim), 1.0),
            (rng.uniform(-5, 5, dim), 0.8),
        ]

        for i, (x0, sigma0) in enumerate(restart_configs):
            if calls_remaining <= 0 or time.time() >= deadline:
                break

            if i == 0:
                alloc = min(calls_remaining, 200)
            elif i == 1:
                alloc = min(calls_remaining, 120)
            else:
                alloc = min(calls_remaining, 80)

            lam_min = 4 + int(3 * np.log(dim))
            if alloc < lam_min:
                alloc = calls_remaining

            bx, bf, used = self._cmaes_run(func, x0, sigma0, alloc, rng, deadline)
            calls_remaining -= used

            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()

            if global_best_f < 1e-6:
                break

        # Final Nelder-Mead polish with remaining budget
        if calls_remaining > 5 and time.time() < deadline - 5.0:
            polish_calls = min(calls_remaining, 50)
            px, pf, used = self._nelder_mead_polish(func, global_best_x, polish_calls, deadline)
            calls_remaining -= used
            if pf < global_best_f:
                global_best_f = pf
                global_best_x = px.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
