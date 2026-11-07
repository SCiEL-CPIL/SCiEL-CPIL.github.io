"""Rewritten optimizer for BBOB f14 (Different powers) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar


class Optimizer:
    """Hybrid CMA-ES style optimizer with local refinement and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # State
        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and call_count < budget

        # Phase 1: Initial exploration with diverse starting points
        n_init = min(10, budget // 10)
        # Latin Hypercube sampling for initial points
        init_points = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            init_points[:, d] = (perm + rng.uniform(size=n_init)) / n_init * 10.0 - 5.0

        for i in range(n_init):
            if not time_ok():
                break
            evaluate(init_points[i])

        # Phase 2: CMA-ES style evolution from best initial point
        if not time_ok():
            return best_x, best_f

        x = best_x.copy()
        sigma = 1.0
        # Covariance matrix
        C = np.eye(dim)
        # Step sizes for C and sigma
        mu = max(1, dim // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Eigendecomposition cache
        D = np.ones(dim)
        B = np.eye(dim)
        last_decomp = 0

        while time_ok():
            # Generate lambda offspring
            lam = 2 * mu
            if lam > budget - call_count:
                lam = max(1, budget - call_count)
                mu = max(1, lam // 2)
                if mu == 0:
                    break
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights ** 2)

            # Update eigendecomposition every 1/dim generations
            gen = call_count // max(1, lam)
            if gen != last_decomp and (gen % max(1, dim // 2) == 0):
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                last_decomp = gen

            # Generate offspring
            offspring = []
            for i in range(lam):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                offspring.append((y, fy))

            if len(offspring) < mu:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])
            selected = offspring[:mu]

            # Weighted mean of selected
            m_new = np.zeros(dim)
            for i, (y, fy) in enumerate(selected):
                m_new += weights[i] * y
            m_new /= np.sum(weights[:mu])

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B.T @ (D * ((m_new - x) / sigma)))
            hsig = np.sqrt((1 - (1 - c_sigma) ** (2 * (call_count + 1))) / (1 - (1 - c_sigma) ** 2)) if (1 - c_sigma) ** (2 * (call_count + 1)) < 1 else 1.0
            if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (dim + 1)) * hsig:
                p_c = (1 - c_c) * p_c + c_c * (m_new - x) / sigma

            # Update covariance
            delta_h = (1 - c_c) ** (2 * (call_count + 1))
            rank_one = np.outer(p_c, p_c) / sigma ** 2
            rank_mu = np.zeros((dim, dim))
            for i, (y, fy) in enumerate(selected):
                diff = (y - x) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)
            C = (1 + c_1 * (c_mu - c_1) - c_1 - c_mu) * C + c_1 * (rank_one + (1 - delta_h) * p_sigma[:, None] * p_sigma[None, :] / (sigma ** 2) * (B @ B.T)) + c_mu * rank_mu

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = m_new

            # Periodic local refinement
            if call_count % 50 == 0 and time_ok():
                for _ in range(3):
                    if not time_ok():
                        break
                    # Nelder-Mead local optimization
                    try:
                        res = minimize(
                            lambda xx: evaluate(np.clip(xx, -5, 5)),
                            best_x,
                            method='Nelder-Mead',
                            options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-8}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = np.clip(res.x, -5, 5)
                    except Exception:
                        pass

        # Final local refinement if time permits
        if time_ok() and best_x is not None:
            try:
                res = minimize(
                    lambda xx: evaluate(np.clip(xx, -5, 5)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
