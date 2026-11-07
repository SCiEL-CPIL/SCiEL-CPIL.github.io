"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import linalg as la


class Optimizer:
    """Hybrid: fast random restarts + (1+1)-ES polish + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        t0 = time.time()
        deadline = t0 + 285.0

        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # --- Phase 1: Global random restarts for fast early progress ---
        # Büche-Rastrigin is multimodal; sample broadly to find a good basin.
        x = clip(rng.uniform(-5, 5, d))
        fx = float(func(x))
        calls += 1
        best_x, best_f = x.copy(), fx

        # 15 random starts to quickly locate a promising basin
        for _ in range(15):
            if calls >= budget or time.time() > deadline:
                break
            x = clip(rng.uniform(-5, 5, d))
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # --- Phase 2: Short (1+1)-ES polish to lock onto the basin ---
        sigma = 1.0
        for _ in range(30):
            if calls >= budget or time.time() > deadline:
                break
            y = clip(best_x + sigma * rng.standard_normal(d))
            fy = float(func(y))
            calls += 1
            if fy < best_f:
                best_f, best_x = fy, y.copy()
                sigma = min(sigma * 1.25, 4.0)
            else:
                sigma = max(sigma * 0.85, 0.01)

        # --- Phase 3: CMA-ES local refinement ---
        mu = best_x.copy()
        C = np.eye(d)
        sigma = max(0.2, 0.5 * min(1.0, np.linalg.norm(mu)))
        n = 6  # lambda

        # Precompute CMA-ES constants
        m = n // 2
        w = np.log(np.arange(1.0, m + 1)) - np.mean(np.log(np.arange(1.0, m + 1)))
        w = w / w.sum()
        mueff = int(1 / np.sum(w[:m] ** 2))
        cc = (4 + mueff / (d + 4)) / ((d + 2) + 2 * mueff / (d + 4))
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        ps = np.zeros(d)
        pc = np.zeros(d)

        while calls < budget and time.time() < deadline:
            # Generate population
            try:
                B = la.cholesky(C + 1e-12 * np.eye(d), lower=True)
            except np.linalg.LinAlgError:
                C = np.eye(d)
                B = np.eye(d)

            xs = []
            fs = []
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(d)
                x = clip(mu + sigma * (B @ z))
                f = float(func(x))
                calls += 1
                xs.append(x)
                fs.append(f)
                if f < best_f:
                    best_f, best_x = f, x.copy()

            if not xs:
                break

            # Sort by fitness
            order = np.argsort(fs)
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]

            # Weighted mean
            new_mu = np.zeros(d)
            for i in range(m):
                new_mu += w[i] * xs[i]
            mu = clip(new_mu)

            # Update path accumulators and covariance
            for i in range(m):
                diff = (xs[i] - mu) / sigma
                ps = (1 - cs) * ps + cs * diff
                ps_norm = np.linalg.norm(ps)
                hsig = 1 if ps_norm / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 2 / (d + 1)) * chi else 0
                pc = (1 - cc) * pc + hsig * cc * (ps - (1 - cs) * pc)
                C = (1 + c1 + cmu - c1) * C
                C += c1 * (np.outer(ps, ps) - C)
                for i in range(m):
                    diff = (xs[i] - mu) / sigma
                    C += cmu * w[i] * (np.outer(diff, diff) - C)

            # Sigma update (CMS)
            sigma *= np.exp(cs / chi * (ps_norm / chi - 1))
            sigma = min(max(sigma, 1e-10), 10.0)

            # Ensure C stays positive definite
            if np.any(np.diag(C) < 1e-10):
                C = np.eye(d)

        return best_x, best_f
# EVOLVE-BLOCK-END
