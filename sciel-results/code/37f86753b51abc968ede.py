"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = None
        best_f = np.inf
        calls_used = 0

        def time_check():
            return time.time() < deadline

        def cma_es(x0, sigma0, max_calls):
            nonlocal calls_used, best_x, best_f
            x = np.array(x0, dtype=float)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            for gen in range(max_calls // lam + 1):
                if not time_check() or calls_used >= max_calls:
                    break

                # Generate population
                ys = np.array([rng.standard_normal(dim) for _ in range(lam)])
                # Apply CMA-ES step
                B = np.linalg.eigh(C)[1]
                D = np.sqrt(np.linalg.eigh(C)[0])
                ys = (ys * D) @ B.T
                ys *= sigma
                pops = x + ys
                # Clip to bounds
                pops = np.clip(pops, -5, 5)

                # Evaluate
                fvals = []
                for i in range(lam):
                    if not time_check() or calls_used >= max_calls:
                        fvals.append(np.inf)
                        continue
                    fi = func(pops[i])
                    calls_used += 1
                    fvals.append(fi)
                    if fi < best_f:
                        best_f, best_x = fi, pops[i].copy()

                fvals = np.array(fvals)
                order = np.argsort(fvals)

                # Update mean
                x_new = np.sum(weights[:, None] * pops[order[:mu]], axis=0)
                # Update covariance
                y = (pops[order[:mu]] - x) / sigma
                delta_sigma = (np.linalg.norm(x_new - x) / (sigma * chi_n) - 1) * damps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_new - x)
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_new - x)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma)
                C = (1 + c_1 * (2 - c_1) - c_mu) * C \
                    + c_1 * np.outer(pc, pc) \
                    + c_mu * np.sum(weights[:, None, None] * np.outer(y.T, y), axis=2)
                if not hsig:
                    C += c_sigma * (2 - c_sigma) * np.outer(ps, ps)

                x = x_new
                sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Check for convergence
                if np.linalg.norm(C, ord='fro') < 1e-10:
                    break

            return x, best_f

        # Multi-restart strategy
        rng = np.random.default_rng(0)

        # Phase 1: Global exploration with larger sigma
        x0 = rng.uniform(-4, 4, dim)
        cma_es(x0, 2.0, min(budget // 2, 250))

        if not time_check() or calls_used >= budget:
            return best_x, best_f

        # Phase 2: Local exploitation around best found
        remaining = budget - calls_used
        x1 = best_x + rng.uniform(-0.5, 0.5, dim)
        x1 = np.clip(x1, -5, 5)
        cma_es(x1, 0.5, min(remaining, 200))

        if not time_check() or calls_used >= budget:
            return best_x, best_f

        # Phase 3: Fine-tuning
        remaining = budget - calls_used
        x2 = best_x + rng.uniform(-0.1, 0.1, dim)
        x2 = np.clip(x2, -5, 5)
        cma_es(x2, 0.1, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
