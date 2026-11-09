"""Portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5.

Strategy: Multi-phase:
1. LHS space-filling for fast early AOCC
2. CMA-ES for robust exploitation
3. Periodic restarts from best to escape local minima
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: LHS + CMA-ES with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 275.0

        rng = np.random.default_rng(42)
        lo, hi = -5.0, 5.0

        best_x = None
        best_f = np.inf
        calls = 0

        def eval_point(x):
            nonlocal calls, best_x, best_f
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---- Phase 1: LHS space-filling exploration ----
        n_lhs = min(40, budget)
        if budget > n_lhs:
            lhs = self._latin_hypercube(n_lhs, dim, rng)
            for x in lhs:
                if calls >= budget or time.time() > deadline:
                    break
                eval_point(x)

        if best_x is None:
            best_x = rng.uniform(lo, hi, dim)
            best_f = eval_point(best_x)

        # ---- Phase 2: CMA-ES exploitation with restarts ----
        remaining = budget - calls
        if remaining <= 0 or time.time() > deadline:
            return best_x, best_f

        # CMA-ES parameters
        pop_size = max(6, int(4 + 3 * np.log(dim)))
        max_iters = remaining // pop_size + 1

        # Track restarts
        restart_count = 0
        restart_threshold = 30  # no improvement after this many iterations
        best_f_cma = best_f
        best_x_cma = best_x.copy()

        for restart in range(10):
            if calls >= budget or time.time() > deadline:
                break

            # Initialize CMA-ES from current best
            x_mean = best_x_cma.copy()
            sigma = 2.0 if restart == 0 else 1.5 + 0.3 * restart
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_sqrt_inv = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            cmac_c = (4 + np.log(dim)) / (dim + 4)
            cmac_mu = min(pop_size // 2, int(0.5 * (dim + 2)))
            weights = np.log(cmac_mu + 0.5) - np.log(np.arange(1, cmac_mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            no_improve = 0
            best_f_this_restart = best_f_cma

            for it in range(max_iters):
                if calls >= budget or time.time() > deadline:
                    break

                # Generate population
                ys = np.empty((pop_size, dim))
                fs = np.empty(pop_size)
                for i in range(pop_size):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Sample from N(mean, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, lo, hi)
                    ys[i] = y
                    fs[i] = eval_point(y)

                if calls < pop_size:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys = ys[order]
                fs = fs[order]

                # Update best
                if fs[0] < best_f_cma:
                    best_f_cma = fs[0]
                    best_x_cma = ys[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= restart_threshold:
                    break

                # Update mean
                x_mean_new = np.sum(weights[:, None] * ys[:cmac_mu], axis=0)
                x_mean = x_mean_new

                # Update sigma, C, pc, ps
                y_vec = (x_mean_new - x_mean) / sigma if sigma > 1e-10 else np.zeros(dim)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt_inv @ y_vec)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1)))) / chiN < (1.4 + 2 / (dim + 1))
                pc = (1 - cc) * pc + hsig * cc * y_vec

                # Update C
                outer = np.outer(pc, pc)
                diff = C - C_inv
                C = (1 + c1 * hsig * (2 - cc)) * C + cmu * np.sum(
                    weights[:cmac_mu][:, None, None] * np.outer(ys[:cmac_mu] - x_mean_new, ys[:cmac_mu] - x_mean_new),
                    axis=0
                ) + c1 * (outer - diff)

                # Regularize
                C = 0.5 * (C + C.T)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1 / eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
                sigma = min(max(sigma, 1e-4), 10.0)

            # After CMA-ES phase, if we found something better, update global
            if best_f_cma < best_f:
                best_f = best_f_cma
                best_x = best_x_cma.copy()

            # If no improvement, try a random restart
            if no_improve >= restart_threshold and calls < budget and time.time() < deadline:
                restart_count += 1
                if restart_count < 5:
                    x_restart = best_x_cma + 3.0 * rng.standard_normal(dim)
                    x_restart = np.clip(x_restart, lo, hi)
                    f_restart = eval_point(x_restart)
                    if f_restart < best_f:
                        best_f = f_restart
                        best_x = x_restart.copy()
                        best_f_cma = f_restart
                        best_x_cma = x_restart.copy()

        return best_x, best_f

    @staticmethod
    def _latin_hypercube(n, dim, rng):
        """Generate n Latin hypercube points in [lo, hi]^dim."""
        pts = np.empty((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            u = rng.uniform(0, 1, n)
            vals = (perm + u) / n
            pts[:, d] = vals
        lo, hi = -5.0, 5.0
        pts = lo + (hi - lo) * pts
        return pts
# EVOLVE-BLOCK-END
