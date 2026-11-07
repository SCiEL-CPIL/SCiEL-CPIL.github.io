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
        deadline = start_time + 290  # safety margin before 330s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_fx = np.inf
        total_calls = 0

        def safe_func(x):
            nonlocal total_calls
            x = np.clip(x, -5, 5)
            val = func(x)
            total_calls += 1
            return val

        # CMA-ES implementation
        def cma_es(x0, sigma, max_calls, rng):
            nonlocal best_x, best_fx, total_calls
            
            d = len(x0)
            # Population size
            lam = int(4 + 3 * np.log(d))
            # Learning rates
            alpha_mu = 1 / (0.5 * lam)
            mu = lam // 2  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / ((d + 2) * (mu_eff + 5))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

            mean = x0.copy()
            C = np.eye(d)
            sqrt_C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            calls = 0

            while calls < max_calls and time_ok():
                # Generate population
                ys = []
                for i in range(lam):
                    if not time_ok() or calls + 1 >= max_calls:
                        break
                    b = rng.standard_normal(d)
                    y = mean + sigma * sqrt_C @ b
                    y = np.clip(y, -5, 5)
                    fy = safe_func(y)
                    ys.append((fy, y))
                    calls += 1

                if len(ys) == 0:
                    break

                # Sort by fitness
                ys.sort(key=lambda x: x[0])
                
                # Update best
                if ys[0][0] < best_fx:
                    best_fx = ys[0][0]
                    best_x = ys[0][1].copy()

                # Compute weighted mean of best mu
                y_w = np.zeros(d)
                for i in range(min(mu, len(ys))):
                    y_w += weights[i] * ys[i][1]

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(sqrt_C @ y_w) * (sqrt_C @ y_w)
                h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1)) < (1 + 0.9) * np.trace(np.eye(d)) / d else 0
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

                # Update covariance matrix
                delta_d = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sigma) * delta_d) + c_mu * np.sum([weights[i] * np.outer(ys[i][1] - mean, ys[i][1] - mean) for i in range(min(mu, len(ys)))], axis=0)

                # Ensure positive definiteness
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                if np.min(eigenvalues) < 1e-10:
                    C = np.eye(d)
                    sqrt_C = np.eye(d)
                else:
                    sqrt_C = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-10))) @ eigenvectors.T

                # Update mean
                mean = y_w

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * d) - 1))
                sigma = np.clip(sigma, 1e-10, 10)

                # Check convergence
                if np.linalg.norm(mean - x0) < 1e-8 * d and sigma < 1e-6:
                    break

            return mean, sigma, calls

        # Phase 1: CMA-ES with multiple restarts
        # Use most of the budget for CMA-ES, keep some for polish
        cma_budget = int(budget * 0.85)
        restarts = 3
        budget_per_restart = cma_budget // restarts

        for r in range(restarts):
            if not time_ok() or total_calls >= budget:
                break

            if r == 0:
                # First restart: start from center
                x0 = np.zeros(d)
                sigma = 2.0
            else:
                # Subsequent restarts: start near best with perturbation
                if best_x is not None:
                    x0 = best_x + 1.0 * rng.standard_normal(d)
                    x0 = np.clip(x0, -5, 5)
                    sigma = 0.5
                else:
                    x0 = rng.uniform(-5, 5, d)
                    sigma = 2.0

            remaining = min(budget_per_restart, budget - total_calls)
            if remaining < 10:
                break

            _, _, _ = cma_es(x0, sigma, remaining, rng)

        # Phase 2: Nelder-Mead polish from best point
        if time_ok() and best_x is not None:
            try:
                from scipy.optimize import minimize

                # Track calls made by Nelder-Mead
                nm_calls = [0]
                remaining_budget = budget - total_calls

                def nm_func(v):
                    if nm_calls[0] >= remaining_budget:
                        return best_fx
                    nm_calls[0] += 1
                    total_calls += 1
                    return func(np.clip(v, -5, 5))

                if remaining_budget > 10:
                    res = minimize(
                        nm_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': remaining_budget, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        if best_x is None:
            # Fallback: should not happen, but just in case
            best_x = np.zeros(d)
            best_fx = func(best_x)
            total_calls += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
