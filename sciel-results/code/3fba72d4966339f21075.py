"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import linalg as la


class Optimizer:
    """Hybrid population-restart + CMA-ES for the multi-modal Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _cma_step(self, mu, C, sigma, rng, n, func, budget, xbest, fbest, calls, deadline):
        """One CMA-ES generation. Returns (mu, C, sigma, xbest, fbest, calls)."""
        d = self.dim
        lam = n
        m = lam // 2
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
        for _ in range(lam):
            if calls >= budget or time.time() > deadline:
                break
            B = la.cholesky(C + 1e-12 * np.eye(d), lower=True)
            z = rng.standard_normal(d)
            x = mu + sigma * (B @ z)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < fbest:
                fbest, xbest = f, x.copy()
        # Re-evaluate population deterministically is avoided; instead recompute via stored? 
        # We need the population values; redo cheaply is not possible without extra calls.
        # Instead we track them during the loop.
        return mu, C, sigma, xbest, fbest, calls

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        t0 = time.time()
        # Safety margin: stop ~20s before the 300s wall-clock limit.
        deadline = t0 + 280.0

        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # --- Phase 1: population-based global random restarts (fast early progress) ---
        # Sample a batch, keep the best few, and do short (1+1)-ES refinements from them.
        pop_size = 24
        x = clip(rng.uniform(-5, 5, d))
        fx = float(func(x))
        calls += 1
        best_x, best_f = x.copy(), fx

        # A few random starts to quickly find a good basin (Büche-Rastrigin is multimodal)
        for _ in range(12):
            if calls >= budget or time.time() > deadline:
                break
            x = clip(rng.uniform(-5, 5, d))
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # Short (1+1)-ES polish from the best to lock onto a basin quickly.
        sigma = 1.0
        for _ in range(40):
            if calls >= budget or time.time() > deadline:
                break
            y = clip(best_x + sigma * rng.standard_normal(d))
            fy = float(func(y))
            calls += 1
            if fy < best_f:
                best_f, best_x = fy, y.copy()
                sigma *= 1.25
            else:
                sigma *= 0.85
            sigma = min(sigma, 4.0)

        # --- Phase 2: CMA-ES local refinement (fast, low per-call overhead) ---
        mu = best_x.copy()
        C = np.eye(d)
        sigma = max(0.3, 1.0 * min(1.0, np.linalg.norm(mu)))
        # Cap remaining budget; use a population that keeps per-call cost low.
        n = 6  # lambda
        # Precompute CMA constants
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
            B = la.cholesky(C + 1e-12 * np.eye(d), lower=True)
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
            mu = new_mu
            # Update step sizes / covariance
            for i in range(m):
                diff = (xs[i] - mu) / sigma
                ps = (1 - cs) * ps + cs * diff
                pc = (1 - cc) * pc + cc * (1 + (1 - cs) * (ps @ C @ ps) / (chi ** 2) - 1) * (ps @ C) / (chi ** 2) if False else (1 - cc) * pc + cc * (ps @ C) / (chi ** 2)
                C = (1 + c1 + cmu - c1) * C
                # rank-one update
                C += c1 * (np.outer(ps, ps) - C)
                # rank-mu update
                for i in range(m):
                    diff = (xs[i] - mu) / sigma
                    C += cmu * w[i] * (np.outer(diff, diff) - C)
            # sigma update (CMS)
            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
            sigma = min(sigma, 10.0)
            # Keep mu in box (CMA-ES is box-agnostic; just clip mean for stability)
            mu = clip(mu)

        return best_x, best_f
# EVOLVE-BLOCK-END
