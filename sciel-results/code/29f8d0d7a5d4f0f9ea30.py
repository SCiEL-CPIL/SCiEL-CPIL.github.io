"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES style population-based optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 320.0  # safety margin before 330s kill

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1)**2 + 2)
        cmu = (2 + 2 / (dim + 1.3) - c1) / ((dim + 1)**2 + 5)
        cs = (dim + 1.7) / (dim + 9.5)
        de = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1)
        cc = (4 + dim / dim) / (4 + dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        def time_ok():
            return (time.time() - start_time) < time_limit

        # Initial point: try a few random starts to handle unknown shift
        rng = np.random.default_rng(0)

        # Quick random search to find a decent starting point (10% of budget)
        n_init = min(10, budget)
        best_x = None
        best_f = np.inf
        for i in range(n_init):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # CMA-ES main loop
        x_mean = best_x.copy()
        sigma = 1.5  # initial step size
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls = n_init

        # Precompute
        for _ in range(budget - n_init):
            if not time_ok():
                break
            if calls >= budget:
                break

            # Sample population
            ys = []
            for i in range(lam):
                if not time_ok() or calls >= budget:
                    break
                zi = rng.standard_normal(dim)
                # Mahalanobis: C is symmetric, use Cholesky-like via eig (cached)
                y = x_mean + sigma * (C @ zi)
                y = np.clip(y, -5, 5)
                ys.append((y, zi))
                calls += 1

            if not ys:
                break

            # Evaluate
            ys_f = []
            for y, zi in ys:
                fy = func(y)
                ys_f.append((fy, y, zi))

            # Track best
            ys_f.sort(key=lambda t: t[0])
            if ys_f[0][0] < best_f:
                best_f = ys_f[0][0]
                best_x = ys_f[0][1].copy()

            # Update
            mu = lam // 2
            mu_eff = 1 / sum(1.0 / (i + 1) for i in range(mu))
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()

            # New mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * ys_f[i][1]
            x_mean = x_new_mean

            # Evolution path
            z_w = np.zeros(dim)
            for i in range(mu):
                z_w += weights[i] * ys_f[i][2]
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (sigma * z_w)
            norm_ps = np.linalg.norm(p_sigma)
            if norm_ps > 0:
                h_sigma = (norm_ps / chiN < (1.4 + 2 / (dim + 1)) * (1 - cs))
            else:
                h_sigma = 1
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * (sigma * z_w)

            # Covariance update
            delta_h = (1 - h_sigma) * cc * (2 - cc)
            C = (1 - c1 - cmu) * C
            for i in range(mu):
                y_i = (ys_f[i][1] - x_new_mean) / sigma
                C += c1 * np.outer(y_i, y_i)
                C += cmu * weights[i] * np.outer(y_i, y_i)
            C += cmu * (1 + delta_h - c1) * np.outer(p_c, p_c)

            # Symmetrize
            C = (C + C.T) / 2

            # Sigma update
            sigma *= np.exp(cs / de * (norm_ps / chiN - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
