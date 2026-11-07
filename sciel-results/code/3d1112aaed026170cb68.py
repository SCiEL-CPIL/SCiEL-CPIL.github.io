"""CMA-ES optimizer for BBOB-style Sphere (f1) with restarts and adaptive scaling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for 5-dimensional sphere optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls = 0

        def make_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial exploration to find a good starting point
        init_points = [np.zeros(n)]
        for _ in range(6):
            init_points.append(rng.uniform(-4, 4, n))
        for i in range(n):
            for val in [2.5, -2.5]:
                pt = np.zeros(n)
                pt[i] = val
                init_points.append(pt)

        init_fxs = []
        for pt in init_points:
            if calls >= self.budget:
                break
            fx = make_call(pt)
            init_fxs.append(fx)

        if init_fxs:
            best_init_idx = int(np.argmin(init_fxs))
            x0 = init_points[best_init_idx].copy()
            f0 = init_fxs[best_init_idx]
        else:
            x0 = np.zeros(n)
            f0 = np.inf

        # Phase 2: CMA-ES
        # CMA-ES parameters
        lambda_ = 10  # Population size
        mu = lambda_ // 2  # Number of selected individuals
        # Weights: log-linear
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 - cc + mueff / (n - 1)) / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial sigma
        sigma = 1.0
        xmean = x0.copy()
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        while calls < self.budget:
            if time.time() > deadline:
                break

            # Check if we should restart CMA-ES
            # Restart if sigma is very small or very large, or if we've made little progress
            if (sigma < 1e-8 or sigma > 50) and calls > 50:
                # Try a new random starting point
                x0_new = best_x.copy() + rng.uniform(-2, 2, n)
                x0_new = np.clip(x0_new, -5, 5)
                f0_new = make_call(x0_new)
                if f0_new < best_f:
                    xmean = x0_new.copy()
                    f0 = f0_new
                else:
                    xmean = best_x.copy()
                    f0 = best_f
                sigma = 1.0
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                continue

            # Generate population
            pop = []
            pop_f = []
            for i in range(lambda_):
                if calls >= self.budget:
                    break
                z = rng.standard_normal(n)
                # Transform using Cholesky of C
                L = np.linalg.cholesky(C)
                y = xmean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = make_call(y)
                pop.append(y)
                pop_f.append(fy)

            if len(pop) < lambda_:
                break

            # Sort by fitness
            indices = np.argsort(pop_f)
            pop_sorted = [pop[i] for i in indices]
            pop_f_sorted = [pop_f[i] for i in indices]

            # Update mean
            xmean_new = np.sum([weights[i] * pop_sorted[i] for i in range(mu)], axis=0)

            # Update covariance matrix
            diff = [pop_sorted[i] - xmean for i in range(mu)]
            delta_mean = xmean_new - xmean
            C = (1 + c1 + cmu - c1) * C
            for i in range(mu):
                C -= c1 * np.outer(diff[i], diff[i])
                C += cmu * weights[i] * np.outer(diff[i], diff[i])

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update step size control variables
            ps = (1 - cs) * ps + cs / chiN * (L @ (xmean_new - xmean) / sigma)
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls - len(pop)))) < (1 + 3 / n) * chiN else 0
            pc = (1 - cc) * pc + hsig * cc * (xmean_new - xmean) / sigma
            C = (1 + c1 + cmu * hsig * (cc - cc)) * C
            C += cmu * hsig * cc * (1 - cc) * np.outer(pc, pc)

            # Update sigma (CSA - Cumulative Step Adaptation)
            sigma *= np.exp(cs / (n + 2) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-12, 100)

            xmean = xmean_new

            # Periodically re-estimate C to avoid numerical issues
            if calls % 100 == 0:
                # Re-estimate covariance from recent points
                recent = pop_sorted[:min(mu, 5)]
                if len(recent) >= 2:
                    C_est = np.cov(np.array(recent).T, bias=True)
                    if C_est is not None and not np.any(np.isnan(C_est)):
                        C = 0.5 * C + 0.5 * C_est
                        C = (C + C.T) / 2

        return best_x, best_f
# EVOLVE-BLOCK-END
