"""Hybrid CMA-ES with Latin Hypercube initialization for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    CMA-ES based optimizer for Katsuura f23.
    1. Latin Hypercube initialization for broad coverage.
    2. CMA-ES for global search with covariance adaptation.
    3. (1+1)-ES polish for final high-precision convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = np.full(dim, -5.0)
        self.lb = np.full(dim, -5.0)
        self.ub = np.full(dim, 5.0)

    def _latin_hypercube(self, n, dim, rng):
        pts = np.empty((n, dim))
        for d in range(dim):
            order = rng.permutation(n)
            pts[order, d] = (np.arange(n) + rng.random(n)) / n
        return self.lb + pts * (self.ub - self.lb)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        dim = self.dim
        lb, ub = self.lb, self.ub
        budget = self.budget

        rng = np.random.default_rng(42)
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError()
            if calls_made >= budget:
                raise TimeoutError()
            x = np.clip(np.asarray(x, dtype=float), lb, ub)
            calls_made += 1
            return func(x)

        # --- Phase 1: Latin Hypercube Initialization ---
        n_lhs = 20
        X_lhs = self._latin_hypercube(n_lhs, dim, rng)
        y_lhs = np.array([safe_func(x) for x in X_lhs])

        idx_best = np.argmin(y_lhs)
        x_best = X_lhs[idx_best].copy()
        f_best = y_lhs[idx_best]

        if calls_made >= budget or time.time() > deadline:
            return x_best, f_best

        # --- Phase 2: CMA-ES ---
        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(dim))
        # Initial sigma
        sigma = 1.0
        # Mean
        m = x_best.copy()
        # Covariance matrix
        C = np.eye(dim)
        # Step sizes
        cs = 1 / (lam + 3)
        ds = 1 + 2 * max(0, (cs - 1) / (lam + 1)) + cs
        # Evolution paths
        ps = np.zeros(dim)
        pz = np.zeros(dim)
        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Budget allocation for CMA-ES
        # Reserve 50 calls for final polish
        cma_budget = budget - calls_made - 50
        cma_generations = cma_budget // lam

        for gen in range(cma_generations):
            if time.time() > deadline - 30.0:
                break
            if calls_made >= budget - 50:
                break

            # Generate population
            population = np.zeros((lam, dim))
            for i in range(lam):
                # Whitening transformation
                B = np.linalg.cholesky(C)
                # Standard normal
                z = rng.standard_normal(dim)
                # Transform
                x_i = m + sigma * B @ z
                x_i = np.clip(x_i, lb, ub)
                population[i] = x_i

            # Evaluate population
            fitness = np.array([safe_func(x) for x in population])

            # Sort by fitness
            order = np.argsort(fitness)
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights**2)

            # Update mean
            m_new = np.sum(weights[:, None] * population[order[:mu]], axis=0)

            # Update evolution paths
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) / sigma * C^(-1/2) * (m_new - m)
            C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C_inv_sqrt @ (m_new - m))

            # Check for divergence
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) > (chi_n + 1.4) * np.sqrt(lam + 1 - 2):
                sigma *= np.exp(-ds / (lam + 3))

            # Update covariance matrix
            # C = (1 - ds - mu_eff * cs * (2 - cs)) * C + cs * mu_eff * (ps @ ps.T + (1 - cs) * (m_new - m) @ (m_new - m).T / sigma**2)
            # Actually, the standard update is:
            # C = (1 - ds - mu_eff * cs * (2 - cs)) * C + cs * ( (1 - cs) * ps @ ps.T + sum(weights_i * (x_i - m) @ (x_i - m).T / sigma**2) )
            # Let's use the simpler, more stable form:
            # C = (1 - ds - mu_eff * cs * (2 - cs)) * C + cs * ( (1 - cs) * ps @ ps.T + sum(weights_i * (x_i - m) @ (x_i - m).T / sigma**2) )
            # But wait, the standard CMA-ES update for C is:
            # C = (1 - ds - mu_eff * cs * (2 - cs)) * C + cs * ( (1 - cs) * ps @ ps.T + sum(weights_i * (x_i - m) @ (x_i - m).T / sigma**2) )
            # This is correct.
            delta_sigma = 1 - (1 - cs) * np.exp(-ds / (lam + 3))
            # Actually, let's just use the standard formula from the CMA-ES paper.
            # C = (1 - ds - mu_eff * cs * (2 - cs)) * C + cs * ( (1 - cs) * ps @ ps.T + sum(weights_i * (x_i - m) @ (x_i - m).T / sigma**2) )
            # Let's compute the sum term
            sum_term = np.zeros((dim, dim))
            for i in range(mu):
                diff = (population[order[i]] - m) / sigma
                sum_term += weights[i] * np.outer(diff, diff)

            C = (1 - ds - mu_eff * cs * (2 - cs)) * C + cs * ((1 - cs) * np.outer(ps, ps) + sum_term)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(dim)

            # Update sigma
            # pz = (1 - ds) * pz + sqrt(ds * (2 - ds) * mu_eff) / sigma * C^(-1/2) * (m_new - m)
            pz = (1 - ds) * pz + np.sqrt(ds * (2 - ds) * mu_eff) / sigma * (C_inv_sqrt @ (m_new - m))

            # Update sigma
            hsig = 1 if np.linalg.norm(pz) / np.sqrt(1 - (1 - ds)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            sigma *= np.exp(cs / lam * (np.linalg.norm(pz) / chi_n - 1) * hsig)
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update mean
            m = m_new

            # Update best
            if fitness[order[0]] < f_best:
                f_best = fitness[order[0]]
                x_best = population[order[0]].copy()

        # --- Phase 3: (1+1)-ES Polish ---
        if calls_made < budget and time.time() < deadline - 5.0:
            sigma = 0.1
            x_es = x_best.copy()
            fx_es = f_best

            while calls_made < budget:
                if time.time() > deadline - 2.0:
                    break

                y_es = x_es + sigma * rng.standard_normal(dim)
                y_es = np.clip(y_es, lb, ub)

                fy_es = safe_func(y_es)

                if fy_es < fx_es:
                    x_es, fx_es = y_es, fy_es
                    sigma *= 1.1
                    if fx_es < f_best:
                        f_best = fx_es
                        x_best = x_es.copy()
                else:
                    sigma *= 0.9

                sigma = np.clip(sigma, 1e-8, 5.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
