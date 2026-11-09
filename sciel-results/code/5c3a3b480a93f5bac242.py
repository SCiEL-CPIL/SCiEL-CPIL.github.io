"""Multi-start CMA-ES + Nelder-Mead polish for BBOB f3 (Rastrigin), with deadline guard."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 300.0  # safety margin before the 330s kill
        calls = [0]

        def f(x):
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = np.reshape(x, (self.dim,))
            calls[0] += 1
            v = func(x)
            v = float(v)
            if not math.isfinite(v):
                v = 1e12
            return v

        rng = np.random.default_rng(12345)

        # ---- diverse starting points (spread across the box) ----
        n_starts = 10
        starts = []
        for k in range(n_starts):
            if k == 0:
                p = np.zeros(self.dim)
            else:
                p = rng.uniform(-4.5, 4.5, self.dim)
            starts.append(p)

        best_x = None
        best_f = float("inf")

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)

        def cmaes(x0, sigma0, budget):
            """CMA-ES. Returns (best_x, best_f)."""
            n = self.dim
            x0 = np.array(x0, dtype=float)
            sigma = float(sigma0)
            mu = max(1, int(math.floor(budget / 6.0)))
            lam = max(2 * mu + 1, 12)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / (w @ w)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            ps = np.zeros(n)
            pc = np.zeros(n)
            B = np.eye(n)
            D = np.ones(n)
            chi_n = math.sqrt(n) * (1 - 1 / (4.0 * n) + 1 / (21.0 * n * n))
            x = np.array(x0, dtype=float)
            fx = f(x)
            update_best(x, fx)
            it = 0
            while it < budget and calls[0] < self.budget and time.time() < deadline:
                # sample
                Y = rng.multivariate_normal(np.zeros(n), np.eye(n), size=lam)
                Y = (B @ (D * Y).T).T
                pops = np.array([x + sigma * yi for yi in Y])
                fs = np.array([f(pops[i]) for i in range(lam)])
                # finite guard
                fs = np.where(np.isfinite(fs), fs, 1e12)
                order = np.argsort(fs)
                wsum = w.sum()
                m = np.sum(w[:, None] * pops[order[:mu]], axis=0) / wsum
                # sigma adaptation
                ps = (1 - c_sigma) * ps + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (B @ (D * ((m - x) / sigma)))
                if np.linalg.norm(ps) < (1.4 + 1 / c_sigma) * chi_n:
                    sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                # covariance adaptation
                y = (m - x) / sigma
                hsig = float(np.linalg.norm(ps) / math.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))))
                hsig = min(1.0, max(0.0, hsig))
                rank_1 = np.outer(y, y)
                rank_mu = np.sum(w[:, None] * np.outer(pops[order[:mu]] - x, pops[order[:mu]] - x), axis=0) / (sigma * sigma)
                D = D * math.exp((c_sigma / (2 * d_sigma)) * (np.linalg.norm(ps) / chi_n - 1))
                B = B + (c_c / (np.linalg.norm(pc) ** 2 + 1e-15) * (pc * (1 + (1 - hsig) * c_sigma) - (1 - c_c) * np.eye(n)))
                pc = (1 - c_c) * pc + c_c * y
                # eigendecomp (guard)
                try:
                    C = (B @ B.T) * (D[:, None] * D[None, :])
                    evals, evecs = np.linalg.eigh(C)
                    evals = np.clip(evals, 1e-20, None)
                    D = np.sqrt(evals)
                    B = evecs
                    # clamp condition number
                    ratio = D.max() / D.min()
                    if ratio > 1e6:
                        D = np.clip(D, D.min() * 1e6, D.max() / 1e6)
                except Exception:
                    pass
                x = m
                fx = fs[order[0]]
                update_best(x, fx)
                it += 1
            return x, fx

        def nelder_mead(x0, budget):
            """Nelder-Mead polish. Returns (best_x, best_f)."""
            n = self.dim
            simplex = [np.array(x0, dtype=float)]
            f0 = f(simplex[0])
            update_best(simplex[0], f0)
            for i in range(n):
                xi = np.array(simplex[0], dtype=float)
                step = 0.5 if abs(xi[i]) < 1e-8 else 0.05 * xi[i]
                xi[i] += step
                simplex.append(xi)
                fi = f(xi)
                update_best(xi, fi)
            # order
            def order_simplex():
                idx = np.argsort([f(s) for s in simplex])
                return [simplex[i] for i in idx]
            it = 0
            while it < budget and calls[0] < self.budget and time.time() < deadline:
                simplex = order_simplex()
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                alpha = 1.0
                xr = centroid + alpha * (centroid - worst)
                fr = f(xr)
                update_best(xr, fr)
                if fr < f(simplex[0]):
                    xe = centroid - alpha * (centroid - worst)
                    fe = f(xe)
                    update_best(xe, fe)
                    simplex[-1] = xe if fe < fr else xr
                elif fr < f(simplex[-2]):
                    simplex[-1] = xr
                else:
                    xc = centroid + 0.5 * (worst - centroid)
                    fc = f(xc)
                    update_best(xc, fc)
                    simplex[-1] = xc
                it += 1
            best_i = np.argmin([f(s) for s in simplex])
            return simplex[best_i], f(simplex[best_i])

        # ---- Phase 1: multi-start CMA-ES ----
        for start in starts:
            if calls[0] >= self.budget or time.time() >= deadline:
                break
            sigma0 = 1.5
            cmaes(start, sigma0, 60)

        # ---- Phase 2: Nelder-Mead polish on the best found ----
        if best_x is not None and calls[0] < self.budget and time.time() < deadline:
            nelder_mead(best_x, 200)

        # ---- Phase 3: a few more short CMA-ES refinements around the best ----
        if best_x is not None and calls[0] < self.budget and time.time() < deadline:
            for _ in range(3):
                if calls[0] >= self.budget or time.time() >= deadline:
                    break
                jitter = best_x + 0.3 * rng.standard_normal(self.dim)
                cmaes(jitter, 0.5, 30)

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = f(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
