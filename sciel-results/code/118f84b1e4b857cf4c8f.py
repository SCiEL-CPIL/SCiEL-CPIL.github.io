"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 20s safety margin
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Restart loop
        for restart in range(5):
            if calls_made >= budget or time.time() > deadline:
                break

            # Initialize
            if best_x is not None and restart > 0:
                x = best_x.copy()
            else:
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            if calls_made >= budget or time.time() > deadline:
                break

            # CMA-ES parameters
            lam = 4 + int(3 * np.log(dim))  # population size
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            sigma = 2.0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            inv_C_sqrt = np.eye(dim)
            # Precompute sqrt of C (identity initially)
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            inv_D = 1.0 / D
            inv_C_sqrt = B @ np.diag(inv_D) @ B.T

            # Track progress for restart decision
            progress_calls = 0
            progress_threshold = 30  # if no improvement in 30 calls, consider restart

            while calls_made < budget and time.time() < deadline:
                # Sample population
                Y = np.zeros((lam, dim))
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    Y[i] = x + sigma * (inv_C_sqrt @ z)
                    Y[i] = np.clip(Y[i], -5, 5)

                # Evaluate
                F = np.zeros(lam)
                for i in range(lam):
                    if calls_made >= budget or time.time() > deadline:
                        F[i] = np.inf
                        break
                    F[i] = func(Y[i])
                    calls_made += 1

                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                idx = np.argsort(F)
                Y_sorted = Y[idx]
                F_sorted = F[idx]

                # Update best
                if F_sorted[0] < best_f:
                    best_f, best_x = F_sorted[0], Y_sorted[0].copy()
                    progress_calls = 0
                else:
                    progress_calls += 1

                # Weighted mean
                w = weights
                y_w = (w[:, None] * Y_sorted[:mu]).sum(axis=0)

                # Step-size control
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_C_sqrt @ y_w / sigma)
                norm_ps = np.linalg.norm(ps)

                # Covariance adaptation
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w / sigma)
                delta_h = (1 - np.exp(-c_sigma / norm_ps * chi_n)) if norm_ps > 1e-12 else 1.0
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - np.exp(-2 * c_c)) * C)
                C += c_mu * (w[:, None] * (Y_sorted[:mu] - x) / sigma) @ ((w[:, None] * (Y_sorted[:mu] - x) / sigma).T)

                # Ensure symmetry
                C = 0.5 * (C + C.T)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 100)

                # Update x
                x = x + y_w
                x = np.clip(x, -5, 5)

                # Update inverse sqrt of C periodically (every 10 iterations for speed)
                if progress_calls % 10 == 0 or progress_calls == 0:
                    try:
                        D, B = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(D, 1e-12))
                        inv_D = 1.0 / D
                        inv_C_sqrt = B @ np.diag(inv_D) @ B.T
                    except:
                        pass

                # Restart if stalled
                if progress_calls > progress_threshold:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
