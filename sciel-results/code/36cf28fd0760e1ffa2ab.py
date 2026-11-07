"""Fresh design: CMA-ES (rotation-adaptive) + adaptive step + restarts + deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Black-box continuous optimizer for a 5-D linear-slope (BBOB f5) task.

    Uses a lightweight CMA-ES (rotation-adaptive via covariance) with an
    adaptive step size, restarts on stagnation, and a wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin vs 300s / 330s kill

        # ---- call accounting (never exceed budget) ----
        calls = [0]
        best_f = [np.inf]
        best_x = [None]

        def count(x):
            if calls[0] >= budget:
                return best_f[0]
            fx = float(func(np.asarray(x, dtype=float)))
            calls[0] += 1
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = np.asarray(x, dtype=float).copy()
            return fx

        # ---- helpers ----
        def randn(n):
            return np.random.randn(n)

        def run_cma(seed, budget_left):
            if budget_left <= 0:
                return
            rng = np.random.RandomState(seed)
            mu = max(2, dim // 2)
            lam = 2 * mu
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = int(np.sum(weights) ** 2 / np.sum(weights ** 2))
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0.0, (mueff - 1) / (dim + 1)) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1.2) ** 2 + mueff)
            c_mu = (2 * (weights[0] - 1) / (dim + 1.0) ** 2 - c_1)
            c_mu = max(0.0, min(c_mu, 0.5 * c_1))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma0 = 1.5

            x = np.random.uniform(-5, 5, dim)
            fx = count(x)
            best_local = fx
            best_local_x = x.copy()

            step = sigma0
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            hs = 1.0
            hsig = (1 + (1 + c_sigma) * np.sqrt((mueff + 1) / dim + 1)) / (1 + c_sigma)
            last_improve = 0

            for g in range(budget_left - 1):
                if time.time() > deadline:
                    return
                if calls[0] >= budget:
                    return

                # adaptive step size (success-rate based)
                if g > 0:
                    step *= np.exp(0.33 * (2.0 * success - 1.0))
                    step = max(1e-4, min(step, 20.0))

                # population
                y = np.empty((lam, dim))
                z = np.empty((lam, dim))
                for i in range(lam):
                    zi = rng.randn(dim)
                    z[i] = zi
                    y[i] = x + step * sigma0 * zi

                fy = np.empty(lam)
                for i in range(lam):
                    fy[i] = count(y[i])
                    if calls[0] >= budget:
                        break

                # track local best (for restart)
                if fy.min() < best_local - 1e-9:
                    best_local = fy.min()
                    best_local_x = y[fy.argmin()].copy()
                    last_improve = g

                # restart on stagnation
                if g - last_improve > 60 and g > 20:
                    return

                # select best mu
                order = np.argsort(fy)
                y_w = y[order[:mu]]
                # weighted mean of z
                mean_z = np.dot(weights, z[order[:mu]])

                # sigma update
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                    c_sigma * (2 - c_sigma) * mueff
                ) / (step * sigma0) * mean_z
                norm_ps = np.linalg.norm(p_sigma)
                if norm_ps > (1.4 + 1 / c_sigma) * chi_n:
                    step *= np.exp(c_sigma * d_sigma * (norm_ps / chi_n - 1))
                    step = max(1e-4, min(step, 20.0))

                # mean update
                y_mean = np.dot(weights, y_w)
                x = y_mean

                # C update
                p_c = (1 - c_c) * p_c + np.sqrt(
                    c_c * (2 - c_c) * mueff
                ) * (y_mean - x)  # note: x already updated above; recompute delta
                # (recompute delta properly)
                p_c = (1 - c_c) * p_c + np.sqrt(
                    c_c * (2 - c_c) * mueff
                ) * (y_mean - y_mean)  # placeholder, fixed below
                return

            return

        # Because the CMA-ES above has a subtle bug risk in the C update,
        # we instead implement a clean, robust variant below.
        # (The function above is intentionally left; we use run_cma2.)

        def run_cma2(seed, budget_left):
            if budget_left <= 0:
                return
            rng = np.random.RandomState(seed)
            mu = max(2, dim // 2)
            lam = 2 * mu
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = int(np.sum(weights) ** 2 / np.sum(weights ** 2))
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0.0, (mueff - 1) / (dim + 1)) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1.2) ** 2 + mueff)
            c_mu = (2 * (weights[0] - 1) / (dim + 1.0) ** 2 - c_1)
            c_mu = max(0.0, min(c_mu, 0.5 * c_1))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            x = np.random.uniform(-5, 5, dim)
            fx = count(x)
            best_local = fx
            best_local_x = x.copy()

            step = 1.5
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            hsig = (1 + (1 + c_sigma) * np.sqrt((mueff + 1) / dim + 1)) / (1 + c_sigma)
            last_improve = 0
            success = 0.5

            for g in range(budget_left - 1):
                if time.time() > deadline:
                    return
                if calls[0] >= budget:
                    return

                # adaptive step size
                if g > 0:
                    step *= np.exp(0.33 * (2.0 * success - 1.0))
                    step = max(1e-4, min(step, 20.0))

                # population (z ~ N(0, C) approximated by Cholesky)
                L = np.linalg.cholesky(C)
                y = np.empty((lam, dim))
                z = np.empty((lam, dim))
                for i in range(lam):
                    zi = rng.randn(dim)
                    z[i] = zi
                    y[i] = x + step * (L @ zi)

                fy = np.empty(lam)
                for i in range(lam):
                    fy[i] = count(y[i])
                    if calls[0] >= budget:
                        break

                # local best + restart
                i_min = int(fy.argmin())
                if fy[i_min] < best_local - 1e-9:
                    best_local = fy[i_min]
                    best_local_x = y[i_min].copy()
                    last_improve = g
                if g - last_improve > 60 and g > 20:
                    return

                # select best mu
                order = np.argsort(fy)
                idx = order[:mu]
                y_w = y[idx]
                z_w = z[idx]
                mean_z = np.dot(weights, z_w)

                # sigma update
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                    c_sigma * (2 - c_sigma) * mueff
                ) * mean_z
                norm_ps = np.linalg.norm(p_sigma)
                if norm_ps > (1.4 + 1 / c_sigma) * chi_n:
                    step *= np.exp(c_sigma * d_sigma * (norm_ps / chi_n - 1))
                    step = max(1e-4, min(step, 20.0))

                # mean update
                y_mean = np.dot(weights, y_w)
                delta = y_mean - x
                x = y_mean

                # C update
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * delta
                rank1 = np.outer(p_c, p_c)
                C = (1 - c_c * (2 - c_c) + c_1 + c_mu) * C + c_1 * (rank1 - C)
                # rank-mu
                for k in range(mu):
                    C += c_mu * np.outer(weights[k] * z_w[k], weights[k] * z_w[k])
                # symmetric
                C = 0.5 * (C + C.T)
                # ensure positive definite
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = 0.9 * C + 0.1 * np.eye(dim)

                # success rate for step adaptation
                success = float(np.mean(fy[idx] < fx))
                fx = fy[i_min]

            return

        # ---- main loop: restarts across the budget ----
        seed = 12345
        base = 80
        while calls[0] < budget and time.time() < deadline:
            b = min(base, budget - calls[0])
            if b < 10:
                b = budget - calls[0]
            if b < 1:
                break
            run_cma2(seed, b)
            seed += 7919
            # shrink restarts as we go to allow more, smaller restarts
            base = max(40, int(base * 0.8))

        # ---- final: if we have spare budget, polish around best with (1+1) ----
        if best_x[0] is not None and calls[0] < budget and time.time() < deadline:
            x = best_x[0].copy()
            fx = best_f[0]
            sigma = 0.5
            for _ in range(budget - calls[0]):
                if calls[0] >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * np.random.randn(dim), -5, 5)
                fy = count(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
