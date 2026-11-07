"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            calls_made += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return func(x)

        def cma_es(initial_x, initial_sigma, budget_for_run):
            """Run a CMA-ES-like optimization for a given budget."""
            nonlocal best_x, best_f, calls_made, deadline

            dim = self.dim
            x = np.array(initial_x, dtype=np.float64)
            fx = safe_call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = initial_sigma
            # CMA-ES parameters
            mu = max(1, dim // 2)  # number of selected individuals
            lambda_ = 2 * dim  # population size
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)

            for gen in range(budget_for_run // lambda_):
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break

                # Generate lambda candidates
                candidates = []
                for i in range(lambda_):
                    if time.time() > deadline or calls_made >= self.budget:
                        break
                    # Sample from N(x, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    # Whitening via Cholesky of C
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    y = x + sigma * L @ z
                    fy = safe_call(y)
                    candidates.append((fy, y))

                if not candidates:
                    break

                # Sort by fitness
                candidates.sort(key=lambda t: t[0])
                selected = candidates[:mu]

                # Update mean
                new_x = np.sum([w * y for w, (_, y) in zip(weights, selected)], axis=0)

                # Update step size control
                old_sigma = sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L @ (np.mean([y for _, y in selected], axis=0) - x) / sigma)
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update evolution path
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (L @ (new_x - x) / sigma)

                # Update covariance
                delta_heav = (1 + 2 * max(0, (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (gen + 1)) - 1) / chi_n - 1)) * c_mu
                outer = np.outer(new_x - x, new_x - x)
                C = (1 - c_1 - c_mu + delta_heav) * C + c_1 * (np.outer(p_c, p_c) + (np.linalg.norm(p_c) ** 2 > (1 - (1 - c_c) ** (gen + 1)) * (2 * chi_n ** 2 - 1) and np.eye(dim) or np.zeros((dim, dim)))) + c_mu / sigma ** 2 * np.sum([w * np.outer(y - x, y - x) for w, (_, y) in zip(weights, selected)], axis=0)

                x = new_x
                fx = safe_call(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            return x, best_f

        # Phase 1: Initial exploration with multiple restarts
        n_restarts = 5
        budget_per_restart = self.budget // (n_restarts + 2)

        for r in range(n_restarts):
            if time.time() > deadline or calls_made >= self.budget:
                break
            # Start from different points
            if r == 0:
                init_x = np.zeros(self.dim)
                init_sigma = 2.0
            else:
                init_x = rng.uniform(-5, 5, self.dim)
                init_sigma = 1.0 + 0.5 * r

            cma_es(init_x, init_sigma, budget_per_restart)

        # Phase 2: Local refinement using scipy minimize
        if best_x is not None and time.time() < deadline and calls_made < self.budget:
            remaining_budget = self.budget - calls_made
            # Use Nelder-Mead for local refinement
            try:
                result = minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining_budget, 200), 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: Final polish with very small steps
        if best_x is not None and time.time() < deadline and calls_made < self.budget:
            sigma = 1e-3
            for _ in range(min(50, self.budget - calls_made)):
                if time.time() > deadline or calls_made >= self.budget:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = safe_call(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
