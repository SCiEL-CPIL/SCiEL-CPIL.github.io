"""Rewritten optimizer: CMA-ES with multi-start and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with multi-start and a hard wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # 10 s safety margin before the 300 s eval limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        # ---------- Phase 1: coarse random sampling ----------
        n_starts = 12
        candidates = rng.uniform(-4.5, 4.5, size=(n_starts, dim))
        # Add a few structured points
        extra = np.array([
            np.zeros(dim),
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            rng.uniform(-4.0, 4.0, size=dim),
        ])
        candidates = np.vstack([candidates, extra])
        n_starts = len(candidates)

        start_xs = []
        start_fs = []
        for i in range(n_starts):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = clip(candidates[i])
            f = func(x)
            evaluated += 1
            start_xs.append(x)
            start_fs.append(f)
            if f < best_f:
                best_f, best_x = f, x.copy()

        # ---------- Phase 2: CMA-ES from the best start ----------
        # Sort starts by fitness
        order = np.argsort(start_fs)
        top_k = min(len(order), 4)

        # Budget for CMA-ES: use most of remaining budget
        cma_budget = budget - evaluated - 20  # leave 20 for polish

        for idx in order[:top_k]:
            if not elapsed_ok() or evaluated >= budget - 20:
                break

            x0 = start_xs[idx].copy()
            f0 = start_fs[idx]
            if f0 >= best_f:
                continue

            # CMA-ES parameters
            lam = max(4, int(4 + 3 * np.log(dim)))  # population size
            sigma = 1.0
            mean = x0.copy()

            # Covariance matrix initialization
            C = np.eye(dim)
            # Step sizes for C and sigma
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = (2 - mu_eff / (dim + 1.8) + cc) / ((dim + 1.6) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            for gen in range(cma_budget // lam + 1):
                if not elapsed_ok() or evaluated >= budget - 20:
                    break

                # Generate population
                ys = []
                fs = []
                for i in range(lam):
                    if not elapsed_ok() or evaluated >= budget - 20:
                        break
                    z = rng.standard_normal(dim)
                    y = clip(mean + sigma * np.linalg.cholesky(C) @ z)
                    fy = func(y)
                    evaluated += 1
                    ys.append(y)
                    fs.append(fy)
                    if fy < best_f:
                        best_f, best_x = fy, y.copy()

                if len(ys) < 2:
                    break

                ys = np.array(ys)
                fs = np.array(fs)

                # Sort by fitness
                idx_sorted = np.argsort(fs)
                ys_sorted = ys[idx_sorted]
                fs_sorted = fs[idx_sorted]

                # Update mean using weighted recombination
                new_mean = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)

                # Update covariance
                y_diff = ys_sorted[:mu] - mean
                delta_sigma = (np.linalg.norm(new_mean - mean) / (sigma * chi_n) - 1) * np.sign(
                    np.linalg.norm(new_mean - mean) > 0
                )
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (
                    (new_mean - mean) / (sigma * chi_n)
                )
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * (
                    (new_mean - mean) / (sigma * chi_n)
                )

                h_sigma = (
                    np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
                    / chi_n
                    < (1.4 + 2 / (dim + 1))
                )

                C = (
                    (1 - c1 - cmu + (1 - c1 - cmu) * cs * (2 - cs) * (not h_sigma)) * C
                    + c1 * (p_c[:, None] @ p_c[None, :] + (not h_sigma) * cs * (2 - cs) * p_sigma[:, None] @ p_sigma[None, :])
                    + cmu * np.sum(weights[:, None, None] * y_diff[:, None, :] * y_diff[:, :, None], axis=0)
                )

                # Ensure symmetry
                C = 0.5 * (C + C.T)

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = float(np.clip(sigma, 1e-8, 10.0))

                mean = new_mean

        # ---------- Phase 3: fine polish with (1+1)-ES ----------
        remaining = budget - evaluated
        if remaining > 0 and best_x is not None and elapsed_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            successes = 0
            window = 0
            WIN = 5
            for _ in range(remaining):
                if not elapsed_ok() or evaluated >= budget:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                evaluated += 1
                window += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                if window >= WIN:
                    if successes / window > 1.0 / 5.0:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    sigma = float(np.clip(sigma, 1e-8, 1.0))
                    successes = 0
                    window = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
