"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            val = func(x)
            calls_made += 1
            return val

        # Phase 1: Multi-start exploration
        rng = np.random.default_rng(42)
        n_starts = 20
        for i in range(n_starts):
            if not check_time() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES refinement from best point
        if best_x is not None and check_time() and calls_made < budget:
            x_best = best_x.copy()
            f_best = best_f
            sigma = 1.0
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = 1.0 / (pop_size + 3)
            c1 = 2 / ((dim + 1) ** 2 + 10)
            cmu = min(1 - c1, 2 * ((dim + 1.7) ** 2 - 1) / ((dim + 2) ** 2 + 13))
            pc_n = np.sqrt((1 + 2 * max(0, (1 + cs) / (1 - cs)) - 1) / (1 - cs))
            cs_n = cs * (1 + 2 * max(0, (1 + cs) / (1 - cs)))

            while check_time() and calls_made < budget:
                # Generate population
                ys = []
                for j in range(pop_size):
                    if not check_time() or calls_made >= budget:
                        break
                    z = rng.standard_normal(dim)
                    y = x_best + sigma * z @ C
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    if fy is None:
                        break
                    ys.append((fy, y, z))

                if len(ys) < pop_size:
                    break

                ys.sort(key=lambda t: t[0])

                # Update best
                if ys[0][0] < f_best:
                    f_best = ys[0][0]
                    best_x = ys[0][1].copy()

                # Compute weighted mean
                weights = np.log(np.arange(1, pop_size + 1))
                weights /= weights.sum()
                mu_eff = 1 / (weights ** 2).sum()
                y_mean = np.zeros(dim)
                for w, (fy, y, z) in zip(weights, ys):
                    y_mean += w * y
                z_mean = np.zeros(dim)
                for w, (fy, y, z) in zip(weights, ys):
                    z_mean += w * z

                # Update evolution path
                pc = (1 - cs) * pc + cs * (z_mean / sigma)
                pc_norm = np.linalg.norm(pc)
                if pc_norm > 1.4 * (1 + cs) * pop_size:
                    pc = pc * (1.4 * (1 + cs) * pop_size / pc_norm)

                # Update covariance matrix
                delta_sigma = (pc_norm / np.sqrt(2 * dim) - (1 - cs)) / (1 - cs)
                C = (1 + c1 * delta_sigma * dim) * C
                C += c1 * (np.outer(pc, pc) - C)
                for w, (fy, y, z) in zip(weights, ys):
                    C += cmu * (np.outer(z, z) - C)

                # Ensure C is positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update mean
                c_sigma = 2 * (mu_eff - 1) / (dim + 1)
                c_mean = 1 + 2 * max(0, np.sqrt(mu_eff / (dim + 1)) - 1)
                x_best = x_best + c_mean * sigma * (y_mean - x_best)
                x_best = np.clip(x_best, -5, 5)

                # Update sigma
                sigma *= np.exp(cs_n * (pc_norm / np.sqrt(2 * dim) - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 20.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
