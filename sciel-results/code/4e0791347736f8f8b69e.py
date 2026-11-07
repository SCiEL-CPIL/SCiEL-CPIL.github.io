"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-to-local strategy: random exploration then CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def run_cmaes(func, x0, sigma0, max_calls, calls_made):
            nonlocal best_x, best_f
            # Parameters for CMA-ES
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)

            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Evaluate initial point
            fx = func(x)
            calls_made += 1
            update_best(x, fx)

            gen = 0
            while calls_made < max_calls:
                if time.time() > deadline:
                    break

                # Generate population
                B = rng.standard_normal((lam, n))

                # C^{1/2} via eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                Y = x + sigma * (B @ sqrt_C.T)

                # Evaluate
                F = []
                Y_evaluated = []
                for i in range(lam):
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    y = np.clip(Y[i], -5, 5)
                    fy = func(y)
                    F.append(fy)
                    Y_evaluated.append(y)
                    calls_made += 1
                    update_best(y, fy)

                if len(F) < mu:
                    break

                F_arr = np.array(F)
                Y_arr = np.array(Y_evaluated)

                # Sort
                idx = np.argsort(F_arr)
                Y_sorted = Y_arr[idx]
                F_sorted = F_arr[idx]

                # Update mean
                x_new = np.sum(w[:mu, np.newaxis] * Y_sorted[:mu], axis=0)

                # Update step size
                y_invC = (Y_sorted[:mu] - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (w[:mu] @ y_invC))

                # Update correlation matrix
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma

                # Update covariance
                norm_ps = np.linalg.norm(ps)
                norm_ps_expected = np.sqrt(n) * (1 - (1 - c_sigma)**(0.5))
                delta_h = 1 - np.exp(-2 + 5 / (n + 1))
                gen += 1
                if norm_ps / np.sqrt(1 - (1 - c_sigma)**((gen + 1) / d_sigma)) < (1.4 + 2 / (n + 1)) * np.trace(C) / n:
                    h_pc = delta_h
                else:
                    h_pc = 0.0

                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
                for i in range(mu):
                    y_i = y_invC[i]
                    C += c_mu * w[i] * np.outer(y_i, y_i)

                C = (C + C.T) / 2

                x = x_new
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))

                # Clip x to box
                x = np.clip(x, -5, 5)

                # Convergence check: if sigma is very small, stop
                if sigma < 1e-10:
                    break

            return best_x, best_f, calls_made

        # Phase 1: Global exploration with 50 uniform random evaluations
        n_explore = 50
        calls_made = 0
        for _ in range(n_explore):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            calls_made += 1
            update_best(x, fx)

        if time.time() > deadline or calls_made >= budget:
            return best_x, best_f

        # Phase 2: Local refinement with CMA-ES from the best point found
        # Use remaining budget for CMA-ES
        remaining = budget - calls_made
        x0 = best_x.copy()
        sigma0 = 1.0  # Reasonable initial step size for local search

        best_x, best_f, calls_made = run_cmaes(func, x0, sigma0, budget, calls_made)

        return best_x, best_f
# EVOLVE-BLOCK-END
