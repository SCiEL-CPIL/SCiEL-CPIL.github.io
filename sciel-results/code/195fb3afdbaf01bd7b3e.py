"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for sharp ridge f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0
        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            calls_made += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(center, sigma, max_iter=300):
            nonlocal calls_made, best_x, best_f
            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            # Learning rates
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / ((n + 1) ** 2 + mu_eff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            # Inverse square root of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_isqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

            best_local = np.inf
            best_local_x = center.copy()

            for iteration in range(max_iter):
                if time.time() > deadline - 3.0:
                    break
                if calls_made >= budget:
                    break

                # Generate population
                pop = np.zeros((lam, n))
                for i in range(lam):
                    z = rng.standard_normal(n)
                    pop[i] = center + sigma * (C_isqrt @ z)

                # Evaluate
                fvals = np.array([safe_call(pop[i]) for i in range(lam)])
                # Handle non-finite
                fvals[~np.isfinite(fvals)] = np.inf

                # Sort
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Update best
                if fvals[0] < best_local:
                    best_local = fvals[0]
                    best_local_x = pop[0].copy()
                if best_local < best_f:
                    best_f = best_local
                    best_x = best_local_x.copy()

                # Check if we're stuck
                if not np.isfinite(fvals[0]):
                    break

                # Weighted mean of best mu individuals
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * pop[i]

                # Update path evolution strategies
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_isqrt @ y_w) / sigma
                norm_p_sigma = np.linalg.norm(p_sigma)
                h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (iteration + 1)) / chi_n < 1.4 + 2 / (n + 1))
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w / sigma

                # Rank-mu update of covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C
                C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * delta_h * np.eye(n))
                for i in range(mu):
                    y_i = C_isqrt @ pop[i]
                    C += c_mu * weights[i] * np.outer(y_i, y_i)

                # Ensure symmetry and positive semi-definite
                C = 0.5 * (C + C.T)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_isqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)

                # Update center
                center = y_w
                center = np.clip(center, -5.0, 5.0)

                # Convergence check
                if np.ptp(fvals[:mu]) < 1e-10 and sigma < 1e-8:
                    break

            return best_local_x, best_local

        # Multi-start CMA-ES
        n_starts = 4
        for s in range(n_starts):
            if time.time() > deadline - 5.0:
                break
            if calls_made >= budget:
                break

            if s == 0:
                center = np.zeros(dim)
                sigma = 2.0
            elif s == 1:
                center = rng.uniform(-3, 3, dim)
                sigma = 1.5
            elif s == 2:
                center = rng.uniform(-5, 5, dim)
                sigma = 1.0
            else:
                # Start near the current best
                center = best_x + rng.normal(0, 0.5, dim)
                sigma = 0.5

            x_opt, f_opt_local = cma_es(center, sigma, max_iter=200)

            if f_opt_local < best_f:
                best_f = f_opt_local
                best_x = x_opt.copy()

        # Final refinement around the best point
        if calls_made < budget - 20 and time.time() < deadline - 10.0:
            x_ref, f_ref = cma_es(best_x, 0.1, max_iter=50)
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
