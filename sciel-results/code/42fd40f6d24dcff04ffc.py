"""Optimizer for BBOB-style f15 (Rastrigin rotated) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for rotated Rastrigin in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        deadline = time.time() + 285

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=float).reshape(dim))
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                return 1e10

        def run_cmaes(start_x, max_calls, sigma0=2.0):
            """Run CMA-ES from start_x with at most max_calls evaluations."""
            nonlocal best_x, best_f

            lam = max(4 * dim, 10)  # population size
            # CMA-ES parameters
            alpha = 1
            c_sigma = (alpha + 1) / ((dim + 1.3) ** 2 + alpha)
            c_c = (4 + dim / dim) / ((dim + 2) ** 2 + 4)
            mu = lam // 2  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_1 = (2 / ((dim + 1.7) ** 2 + mu_eff)) * (2 - c_c)
            c_mu = min(1 - c_1,
                       (2 * (mu_eff - 2 / 3) / ((dim + 2) ** 2 + mu_eff)) *
                       (2 - c_c))

            x_mean = np.clip(start_x, -5, 5).copy()
            sigma = sigma0
            C = np.eye(dim)
            C_inv = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Initial evaluation
            fx_mean = safe_func(x_mean)
            if fx_mean < best_f:
                best_f = fx_mean
                best_x = x_mean.copy()

            used = 1

            while used < max_calls and call_count < budget and time.time() < deadline:
                # Generate population
                Y = np.zeros((lam, dim))
                F = np.zeros(lam)
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    Y[i] = x_mean + sigma * (C @ z)
                    Y[i] = np.clip(Y[i], -5, 5)
                    F[i] = safe_func(Y[i])
                    used += 1
                    if call_count >= budget or time.time() >= deadline:
                        break

                if call_count >= budget or time.time() >= deadline:
                    break

                # Sort by fitness
                order = np.argsort(F)
                Y = Y[order]
                F = F[order]

                # Update best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = Y[0].copy()

                # Compute new mean
                x_new = np.sum(weights[:, None] * Y[:mu], axis=0)

                # Update path
                C_sqrt = np.linalg.cholesky(C) + 1e-9 * np.eye(dim)
                C_sqrt_inv = np.linalg.inv(C_sqrt)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                     (C_sqrt_inv @ (x_new - x_mean) / sigma)
                norm_ps = np.linalg.norm(ps)

                if norm_ps / chi_n < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (used + 1))):
                    pc = (1 - c_c) * pc + c_c * (x_new - x_mean) / sigma

                # Update covariance
                h_sigma = 1.0 if norm_ps / chi_n < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2) else 0.0
                delta = (1 - h_sigma) * c_c * (1 - c_c)

                C = (1 + c_1 + c_mu - c_1 - delta) * C + \
                    c_1 * np.outer(pc, pc) + \
                    c_mu * np.sum(weights[:, None, None] * np.outer(Y[:mu] - x_mean, Y[:mu] - x_mean) / sigma ** 2, axis=0)

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    C_inv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    C_inv = np.eye(dim)

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                x_mean = x_new

                # Check convergence
                if np.linalg.norm(C, ord='fro') < 1e-12:
                    break

            return x_mean, F[0] if len(F) > 0 else fx_mean

        # Phase 1: CMA-ES from origin with larger sigma
        if call_count < budget and time.time() < deadline:
            start1 = np.zeros(dim)
            budget1 = int(budget * 0.40)
            run_cmaes(start1, budget1, sigma0=2.5)

        # Phase 2: CMA-ES from a random point
        if call_count < budget and time.time() < deadline:
            start2 = rng.uniform(-3, 3, dim)
            budget2 = int(budget * 0.30)
            run_cmaes(start2, budget2, sigma0=2.0)

        # Phase 3: Polish with smaller sigma from best found
        if call_count < budget and time.time() < deadline:
            if best_x is not None:
                start3 = best_x.copy()
            else:
                start3 = rng.uniform(-5, 5, dim)
            budget3 = budget - call_count
            run_cmaes(start3, budget3, sigma0=0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
