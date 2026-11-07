"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_fx = np.inf
        calls = 0

        def safe_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            x = np.clip(x, -5, 5)
            calls += 1
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES
        # Use about 70% of budget for CMA-ES
        cma_budget = int(budget * 0.70)

        # Initialize CMA-ES parameters
        x0 = rng.uniform(-5, 5, d)
        sigma = 2.0

        # CMA-ES implementation
        def cma_es(func_wrapper, x0, sigma, max_calls):
            nonlocal calls, best_x, best_fx

            n = len(x0)
            # Population size
            lam = int(4 + 3 * np.log(n))
            # Weights
            mu = lam // 2
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Adaptation rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # Covariance matrix
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)
            D = 1.0

            # Step size adaptation
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Evolution path for sigma
            p_sigma = np.zeros(n)
            # Evolution path for covariance
            p_c = np.zeros(n)

            # Initial population
            mean = x0.copy()
            best_local_x = mean.copy()
            best_local_fx = func_wrapper(mean)

            remaining = max_calls - 1  # already used 1 call for initial
            it = 0

            while remaining > 0 and it < 10000:
                if not time_ok():
                    break

                # Generate population
                pops = []
                for i in range(lam):
                    if calls + 1 > max_calls + 1:
                        break
                    z = rng.standard_normal(n)
                    y = mean + sigma * (D * C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    pops.append(y)

                if not pops:
                    break

                # Evaluate population
                fvals = []
                for y in pops:
                    if calls >= budget:
                        fvals.append(best_fx)
                    else:
                        fvals.append(func_wrapper(y))
                fvals = np.array(fvals)

                # Sort by fitness
                idx = np.argsort(fvals)
                pops = [pops[i] for i in idx]
                fvals = fvals[idx]

                # Update best
                if fvals[0] < best_local_fx:
                    best_local_fx = fvals[0]
                    best_local_x = pops[0].copy()

                # Compute weighted mean of best mu individuals
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * pops[i]

                # Update evolution path for sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (new_mean - mean) / sigma)

                # Update mean
                mean = new_mean

                # Update evolution path for covariance
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (new_mean - mean + p_c))

                # Compute h_sigma for rank-one update
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))) < (1.4 + 2 / (n + 1)) * chi_n

                # Update covariance matrix
                rank_one = np.outer(p_c, p_c)
                if h_sigma:
                    rank_one += (2 - c_c) * np.outer(p_sigma, p_sigma)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = C_inv_sqrt @ (pops[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 + c_1 * (rank_one - C) + c_mu * (rank_mu - C)) * (1 - c_1 - c_mu)

                # Symmetrize
                C = (C + C.T) / 2

                # Update D, C_sqrt, C_inv_sqrt
                # Simple Cholesky-based update
                try:
                    L = np.linalg.cholesky(C)
                    D = np.prod(np.diag(L)) ** (1.0 / n)
                    C_sqrt = L
                    C_inv_sqrt = np.linalg.inv(L)
                    C_inv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    # Reset if Cholesky fails
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    C_sqrt = np.eye(n)
                    C_inv_sqrt = np.eye(n)
                    D = 1.0

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(1e-8, min(sigma, 10.0))

                remaining -= lam
                it += 1

            return best_local_x, best_local_fx

        # Run CMA-ES
        if time_ok() and calls < cma_budget:
            try:
                cma_x, cma_fx = cma_es(safe_func, x0, sigma, cma_budget)
                if cma_fx < best_fx:
                    best_fx = cma_fx
                    best_x = cma_x
            except Exception:
                pass

        # Phase 2: Nelder-Mead polish from best point
        if time_ok() and best_x is not None:
            try:
                from scipy.optimize import minimize

                nm_calls = [0]

                def nm_func(v):
                    nonlocal calls
                    if nm_calls[0] >= budget - calls:
                        return best_fx
                    nm_calls[0] += 1
                    return func(np.clip(v, -5, 5))

                polish_budget = budget - calls
                if polish_budget > 10:
                    res = minimize(
                        nm_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': polish_budget, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    calls += nm_calls[0]
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
