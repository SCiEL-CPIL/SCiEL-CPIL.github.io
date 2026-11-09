# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start + CMA-ES with adaptive restart for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        DEADLINE = t0 + 295.0
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # --- Phase 1: Latin-hypercube sampling for diverse starts ---
        def _lhs_samples(k, lo, hi, rng):
            u = rng.random((k, n))
            # stratify each dimension
            u = (np.arange(k)[:, None] + u) / k
            return lo + u * (hi - lo)

        # Initial broad sampling
        k0 = 30
        samples = _lhs_samples(k0, -4.5, 4.5, np.random.default_rng(12345))
        for i in range(len(samples)):
            if calls_made >= budget or time.time() > DEADLINE:
                break
            x = samples[i]
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 2: Adaptive CMA-ES with restarts ---
        # We'll run CMA-ES starting from the best point, with restart logic.
        # Since we only have 500 calls, we need to be efficient.
        # Use a simple (1+1)-ES with adaptive step, then switch to CMA-ES for refinement.

        # Actually, let's use a proper CMA-ES with small lambda for 5D.
        # lambda=8, mu=4 for n=5 is reasonable.

        def run_cma_es(x_init, sigma_init, remaining_budget, max_time):
            nonlocal best_x, best_f, calls_made

            lambda_ = 8
            mu_ = 4
            weights = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()

            c_sigma = (mu_eff + 2) / (n + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 - mu_eff + c_sigma) / (n + 4.3)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            x = x_init.copy()
            sigma = sigma_init
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Cholesky cache
            L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            L_inv = np.linalg.inv(L)

            stagnant = 0
            best_in_run = best_f
            best_x_run = best_x.copy() if best_x is not None else x.copy()

            while calls_made < budget and time.time() < max_time:
                # Generate lambda_ offspring
                ys = []
                fs = []
                for j in range(lambda_):
                    if calls_made >= budget or time.time() > max_time:
                        break
                    z = np.random.default_rng(hash((calls_made, j)) & 0xFFFFFFFF).standard_normal(n)
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    ys.append(y)
                    fs.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    if fy < best_in_run:
                        best_in_run = fy
                        best_x_run = y.copy()

                if len(ys) < mu_:
                    break

                # Sort
                order = np.argsort(fs)
                ys_sorted = np.array([ys[i] for i in order])
                fs_sorted = np.array([fs[i] for i in order])

                # Update mean
                x_old = x.copy()
                x_new = np.sum(weights[:mu_][:, None] * ys_sorted[:mu_], axis=0)

                # Check for progress
                if best_in_run < best_f + 1e-10:
                    stagnant = 0
                else:
                    stagnant += 1

                # Update CMA-ES state
                diff = x_new - x_old
                try:
                    v = L_inv @ (diff / sigma)
                except np.linalg.LinAlgError:
                    v = np.zeros(n)

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * v
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                # Update p_c
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * v

                # Update covariance
                y_diffs = (ys_sorted[:mu_] - x_old).T
                try:
                    y_diffs_L_inv = L_inv @ y_diffs
                except np.linalg.LinAlgError:
                    y_diffs_L_inv = np.zeros((n, mu_))

                rank1 = np.outer(p_c, p_c)
                rankmu = np.sum(weights[:mu_][:, None, None] *
                               np.einsum('i,jk,il->ikl', y_diffs_L_inv.T,
                                         np.einsum('jk,kl->jl', y_diffs_L_inv, y_diffs_L_inv.T)),
                               axis=0)
                # Simpler: rankmu = sum w_i * outer(d_i, d_i) where d_i = L_inv @ (y_i - x_old)
                rankmu = np.zeros((n, n))
                for i in range(mu_):
                    d_i = y_diffs_L_inv[:, i]
                    rankmu += weights[i] * np.outer(d_i, d_i)

                C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
                C = (C + C.T) / 2

                x = x_new

                # Update Cholesky
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                    L_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    C += 1e-6 * np.eye(n)
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                    L_inv = np.linalg.inv(L)

                # Restart condition: no improvement for many iterations or sigma too small/large
                if stagnant > 60 or sigma < 1e-8 or sigma > 50:
                    break

            return best_x_run, best_in_run, calls_made

        # Run CMA-ES from the best point found in Phase 1
        if best_x is not None and calls_made < budget and time.time() < DEADLINE:
            remaining = budget - calls_made
            # Use a decent initial sigma
            sigma_init = 1.0
            bx, bf, _ = run_cma_es(best_x, sigma_init, remaining, DEADLINE)

        # --- Phase 3: If we have budget left, try a few more local refinements ---
        if calls_made < budget and time.time() < DEADLINE:
            # Simple coordinate descent refinement
            x = best_x.copy()
            fx = best_f
            step = 0.5
            for coord in range(n):
                if calls_made >= budget or time.time() > DEADLINE:
                    break
                while step > 1e-6 and calls_made < budget and time.time() < DEADLINE:
                    x_plus = x.copy()
                    x_plus[coord] = min(x[coord] + step, 5.0)
                    f_plus = func(x_plus)
                    calls_made += 1

                    x_minus = x.copy()
                    x_minus[coord] = max(x[coord] - step, -5.0)
                    f_minus = func(x_minus)
                    calls_made += 1

                    if f_plus < fx:
                        x = x_plus
                        fx = f_plus
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    elif f_minus < fx:
                        x = x_minus
                        fx = f_minus
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
