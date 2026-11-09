"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with adaptive restarts for f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            calls_made += 1
            f = func(x_clipped)
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # CMA-ES parameters
        def cma_es(x0, sigma0, max_calls, deadline):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget - 10 or time.time() > deadline:
                return

            n = self.dim
            # Population size
            lambda_pop = max(4, int(4 + 3 * np.log(n)))
            # CMA-ES weights
            mu = lambda_pop // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_1 = (2 + 2 * min(1, (mu_eff - 1) / (n + 1))) / (
                (n + 1.3) ** 2 + mu_eff
            )
            c_mu = min(
                1 - c_1,
                2 * (
                    (mu_eff - 2 + 1 / mu_eff) / (n + 2) ** 2
                ) * min(1, c_sigma)
            )

            # State
            mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Evaluate initial mean
            f_mean = safe_call(mean)

            max_iter = max(1, (budget - calls_made - 10) // lambda_pop)
            for it in range(max_iter):
                if calls_made >= budget - 10 or time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_pop):
                    z = rng.standard_normal(n)
                    x = mean + sigma * (B @ (D * z))
                    f_val = safe_call(x)
                    offspring.append((f_val, x, z))

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])

                # Update mean
                new_mean = np.sum(
                    [weights[i] * offspring[i][1] for i in range(mu)], axis=0
                )

                # Update step size (CSA)
                # Simplified: use the old update rule
                z_w = (new_mean - mean) / (sigma * chi_n) if sigma > 0 else np.zeros(n)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w
                h_sigma = (
                    np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (it + 1))
                    / chi_n
                    < 1.4 + 2 / (n + 1)
                )
                if h_sigma:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update covariance
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(
                    c_c * (2 - c_c) * mu_eff
                ) * (new_mean - mean) / sigma

                # Rank-1 and rank-mu updates
                delta_pc = (
                    h_sigma / c_c
                    * (1 - (1 - c_c) ** (it + 1))
                    * pc
                    / sigma
                )
                B, D_inv = np.linalg.eigh(
                    (1 - c_1 - c_mu) * np.eye(n)
                    + c_1 * (np.outer(pc, pc) + delta_pc * delta_pc - delta_pc)
                    + c_mu
                    * np.sum(
                        [
                            weights[i]
                            * np.outer(
                                (offspring[i][1] - mean) / sigma,
                                (offspring[i][1] - mean) / sigma,
                            )
                            for i in range(mu)
                        ],
                        axis=0,
                    )
                )
                D = 1.0 / np.sqrt(np.clip(np.diag(D_inv), 1e-20, None))
                mean = new_mean

        # Multi-start CMA-ES with different initializations
        # Start 1: origin
        cma_es(np.zeros(n), 2.0, budget, deadline)

        # If we still have budget, try a few more starts
        if calls_made < budget - 20 and time.time() < deadline:
            for _ in range(5):
                if calls_made >= budget - 10 or time.time() > deadline:
                    break
                x0 = rng.uniform(-3, 3, n)
                cma_es(x0, 1.5, budget, deadline)

        # Final polish with L-BFGS-B if budget remains
        if best_x is not None and calls_made < budget - 5 and time.time() < deadline:
            def objective(x):
                nonlocal calls_made
                if calls_made >= budget:
                    return np.inf
                calls_made += 1
                return func(np.clip(x, -5, 5))

            try:
                res = minimize(
                    objective,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 50, 'maxfun': 20, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Ensure at least one call was made
        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
