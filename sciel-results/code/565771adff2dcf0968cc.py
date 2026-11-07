"""Multi-start CMA-ES optimizer for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES for Katsuura with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        call_count = [0]
        best_x = [None]
        best_f = [float('inf')]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > self.budget:
                return best_f[0]
            fx = func(np.clip(x, -5.0, 5.0))
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = np.array(x, dtype=float).copy()
            return fx

        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Diverse initial sampling
        initial_points = []
        for _ in range(6):
            initial_points.append(rng.uniform(-4.5, 4.5, dim))
        for sigma in [2.0, 1.0, 0.5]:
            initial_points.append(np.clip(sigma * rng.standard_normal(dim), -4.5, 4.5))
        for _ in range(3):
            initial_points.append(rng.choice([-3.0, -1.0, 1.0, 3.0], size=dim))

        evaluated_starts = []
        for x in initial_points:
            if call_count[0] >= self.budget:
                break
            fx = counted_func(x)
            evaluated_starts.append((fx, np.array(x, dtype=float).copy()))

        evaluated_starts.sort(key=lambda t: t[0])
        remaining = self.budget - call_count[0]

        # Phase 2: CMA-ES from top starts
        n_starts = min(len(evaluated_starts), 4)
        if n_starts >= 4:
            fractions = [0.40, 0.25, 0.20, 0.15]
        elif n_starts == 3:
            fractions = [0.45, 0.30, 0.25]
        elif n_starts == 2:
            fractions = [0.55, 0.45]
        else:
            fractions = [1.0] if n_starts == 1 else []

        for i, (fx_s, x_s) in enumerate(evaluated_starts[:n_starts]):
            if i >= len(fractions):
                break
            if call_count[0] >= self.budget:
                break
            if time.time() > deadline - 15.0:
                break

            nm_budget = int(remaining * fractions[i])
            nm_budget = max(nm_budget, 20)
            nm_budget = min(nm_budget, self.budget - call_count[0])
            if nm_budget < 15:
                break

            x_best, f_best = self._cma_es(counted_func, x_s, nm_budget, deadline)
            if f_best < best_f[0]:
                best_f[0] = f_best
                best_x[0] = np.array(x_best, dtype=float).copy()
            remaining = self.budget - call_count[0]
            if remaining < 15:
                break

        return best_x[0], best_f[0]

    def _cma_es(self, func, x0, budget, deadline):
        """Lightweight CMA-ES implementation."""
        dim = len(x0)
        x0 = np.array(x0, dtype=float)
        local_rng = np.random.default_rng(0)

        best_x = x0.copy()
        best_f = func(x0)
        calls_used = 1

        pop_size = max(dim + 3, 8)
        sigma = 0.8
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        mu = pop_size // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (1 - 1 / (4 * dim))
        cmu = min(1 - c1, (2 * (mueff - 2 + 2 / mueff)) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while calls_used < budget:
            if time.time() > deadline - 5.0:
                break

            # Generate population
            y = local_rng.standard_normal((pop_size, dim))
            x_pop = np.array([x0 + sigma * (B @ (D * y[i])) for i in range(pop_size)])
            x_pop = np.clip(x_pop, -5.0, 5.0)
            f_pop = np.array([func(xi) for xi in x_pop])
            calls_used += pop_size

            # Sort
            order = np.argsort(f_pop)
            x_pop = x_pop[order]
            f_pop = f_pop[order]

            if f_pop[0] < best_f:
                best_f = f_pop[0]
                best_x = x_pop[0].copy()

            # Update mean
            m_new = np.sum(w[:, None] * x_pop[:mu], axis=0)
            x0 = m_new

            # Update step size and covariance
            y_mean = (m_new - x0) / sigma if sigma > 1e-10 else np.zeros(dim)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ (D ** -1)) @ y_mean
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used / pop_size + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                pc = (1 - cs) * pc + cs / damps * (y_mean - (1 - cs) * pc)

            # Covariance update
            delta_h = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            rank_mu = np.sum(w[:, None, None] * np.outer(x_pop[:mu] - m_new, x_pop[:mu] - m_new), axis=0)
            C = (1 + c1 * delta_h + cmu - 1 - c1) * C + c1 * np.outer(y_mean, y_mean) + cmu / sigma ** 2 * rank_mu

            # Decompose C
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                break

            # Adapt sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Convergence check
            if np.max(D) / np.min(D) > 1e7 or sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
