"""Optimizer for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start CMA-ES with a global-scan warm start and a wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, max_calls, deadline):
        """Small CMA-ES. Returns (best_x, best_f, calls_used)."""
        n = self.dim
        rng = np.random.default_rng()
        x = np.array(x0, dtype=float)
        fx = float(func(x))
        calls = 1
        sigma = float(sigma0)
        C = np.eye(n)
        Cinv = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        mu = max(1, n // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4.0 + mueff / n) / ((n + 2.0) ** 2 + mueff / 2.0)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = (2.0 / ((n + 1.8) ** 2 + mueff)) * (2.0 - cc + cs)
        cmu = min(1.0 - c1, (2.0 * (mueff - 2.0 + 1.0 / mueff)) /
                 ((n + 2.0) ** 2 + mueff))
        chi = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
        while calls < max_calls and time.time() < deadline:
            batch = []
            for i in range(mu):
                z = rng.standard_normal(n)
                y = x + sigma * z
                batch.append((y, z, w[i]))
            ys = np.array([b[0] for b in batch])
            zs = np.array([b[1] for b in batch])
            ws = np.array([b[2] for b in batch])
            fvals = np.array([float(func(y)) for y in ys])
            calls += mu
            order = np.argsort(fvals)
            ys = ys[order]
            fvals = fvals[order]
            zs = zs[order]
            xmean = np.sum(ws[:, None] * ys, axis=0)
            # update ps, pc, C
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv @ (zs.mean(axis=0, keepdims=True) * np.sqrt(mueff)).T[0]
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls + 1)))) < (1.4 + 2.0 / (n + 1)) * chi
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc)) * (xmean - x) / sigma
            delta = (pc @ C @ pc) / (pc @ pc + 1e-300)
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C) \
                + cmu * np.sum(ws[:, None, None] * np.outer(ys - x, ys - x), axis=0) / (sigma ** 2 + 1e-300)
            # recompute inverse (n small)
            Cinv = np.linalg.inv(C + 1e-12 * np.eye(n))
            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1.0))
            x = xmean
            fx = float(fvals[0])
            # sigma adaptation guard
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0
            if sigma < 1e-12:
                break
        return x, fx, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            calls_used += 1
            return float(func(np.clip(np.asarray(x, dtype=float), -5.0, 5.0)))

        # ---- Phase 1: global scan (cheap, gets early progress) ----
        n_scan = min(budget, 150)
        for _ in range(n_scan):
            x = rng.uniform(-5.0, 5.0, dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            if calls_used >= budget or time.time() > deadline:
                break

        # ---- Phase 2: multi-start CMA-ES from best points ----
        # Build a set of starting points: the global best plus a few random ones.
        starts = [best_x.copy()]
        for _ in range(3):
            starts.append(rng.uniform(-5.0, 5.0, dim))

        for s in starts:
            if calls_used >= budget or time.time() > deadline:
                break
            # give each CMA-ES run a fair share of the remaining budget
            remaining = budget - calls_used
            if remaining < 20:
                break
            run_budget = max(20, remaining // (len(starts) + 1))
            # sigma: scale by remaining search radius; start moderately wide
            sigma0 = 1.5
            try:
                x, fx, used = self._cmaes(safe_func, s, sigma0, run_budget, deadline)
            except Exception:
                break
            calls_used += used
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # ---- Phase 3: final polish from the global best if budget remains ----
        remaining = budget - calls_used
        if remaining >= 20 and best_x is not None and time.time() < deadline:
            try:
                x, fx, used = self._cmaes(safe_func, best_x, 0.5, remaining, deadline)
                calls_used += used
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
            except Exception:
                pass

        # safety: ensure we don't exceed budget
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
