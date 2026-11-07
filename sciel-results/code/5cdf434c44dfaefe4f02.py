"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive population and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_used
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start: try a few starting points
        rng = np.random.default_rng(42)
        n_starts = 3
        for start_i in range(n_starts):
            if calls_used >= budget or time.time() >= deadline:
                break

            # Initialize CMA-ES
            if start_i == 0:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Perturb from best found so far or random
                if best_x is not None and start_i == 1:
                    x0 = np.clip(best_x + rng.normal(0, 1, dim), -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)

            # CMA-ES parameters
            sigma = 1.0
            pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~11 for dim=5
            # Adapt pop_size: start smaller, grow if budget allows
            pop_size = min(pop_size, 15)

            # CMA-ES state
            mean = x0.copy()
            C = np.eye(dim)
            # Eigen decomposition for step
            eigvals, eigvecs = np.linalg.eigh(C)
            # Inverse square root
            eigvals_inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, 1e-12))
            C_inv_sqrt = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T

            # Evolution path
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            cs = (2 + 2 / (dim + 1)) / (dim + 2 + 2 / (dim + 1))
            cmu = 2 / ((dim + 1.3) ** 2 + pop_size)
            cc = (4 + dim / (dim + 4)) / ((dim + 2) + 4 + dim / (dim + 4))
            damps = (1 + 2 * max(0, np.sqrt((pop_size / (dim + 1) - 1) * (cs - 1))) + cs)
            # Update rules
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Budget for this restart
            max_iter_this = max(10, (budget - calls_used) // (n_starts - start_i))

            for it in range(max_iter_this):
                if calls_used >= budget or time.time() >= deadline:
                    break

                # Sample population
                for i in range(pop_size):
                    if calls_used >= budget or time.time() >= deadline:
                        break
                    # Standard normal via Cholesky-like using C_inv_sqrt
                    z = rng.standard_normal(dim)
                    x_sample = np.clip(mean + sigma * (C_inv_sqrt @ z), -5, 5)
                    f_i = evaluate(x_sample)
                    if i == 0:
                        xs = [x_sample]
                        fs = [f_i]
                    else:
                        xs.append(x_sample)
                        fs.append(f_i)

                if len(fs) < 2:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                xs_sorted = [xs[i] for i in order]
                fs_sorted = [fs[i] for i in order]

                # Lambda is pop_size, take top mu = pop_size//2
                mu = max(1, pop_size // 2)
                weights = np.ones(mu)
                weights = weights / weights.sum()
                mueff = 1.0 / np.sum(weights ** 2)

                # Update mean
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * xs_sorted[i]
                new_mean = np.clip(new_mean, -5, 5)

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (new_mean - mean) / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (new_mean - mean) / sigma
                c1 = (mueff + 2) / ((dim + 2) ** 2 + mueff)
                cmu_cma = cmu * (2 - c1)
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    yi = (xs_sorted[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(yi, yi)
                C = (1 - c1 - cmu_cma) * C + c1 * rank_one + cmu_cma * rank_mu

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update mean
                mean = new_mean

                # Update C_inv_sqrt periodically
                if it % 5 == 0:
                    # Ensure positive definite
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
                    C_inv_sqrt = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T

                # Check convergence
                if sigma < 1e-10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
