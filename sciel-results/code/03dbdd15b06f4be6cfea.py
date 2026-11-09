# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial import Delaunay


class Optimizer:
    """Surrogate-assisted multi-objective Bayesian optimizer (GP + EHVI)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _pareto(self, F):
        m = F.shape[0]
        keep = np.zeros(m, dtype=bool)
        for i in range(m):
            b = (F <= F[i]).all(axis=1)
            if not b.any():
                keep[i] = True
        if not keep.any():
            keep = np.ones(m, dtype=bool)
        return keep

    def _hypervolume(self, F, ref):
        F = np.asarray(F, dtype=float)
        ref = np.asarray(ref, dtype=float)
        if F.size == 0:
            return 0.0
        F = F[(F < ref).all(axis=1)]
        if F.shape[0] == 0:
            return 0.0
        F = F[self._pareto(F)]
        if F.shape[0] == 0:
            return 0.0
        F = F[np.lexsort(F[:, ::-1])]
        hv = 0.0
        prev = ref[0]
        for i in range(F.shape[0]):
            hv += (prev - F[i, 0]) * (ref[1] - F[i, 1])
            prev = F[i, 0]
        return hv

    def _gp(self, X, Y, n):
        d = X.shape[1]
        ell = np.clip(0.02 + 0.05 * np.abs(X).max() / max(X.ptp(), 1e-9), 1e-3, 0.5)
        sig = max(Y.std() * 1.0, 1e-6)
        noise = max(Y.std() * 1e-3, 1e-6)
        K = sig * sig * np.exp(-0.5 * ((X[:, :, None] - X[:, None, :]) ** 2).sum(1) / ell ** 2)
        K = K + np.eye(n) * noise
        try:
            L = np.linalg.cholesky(K + 1e-9 * np.eye(n))
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-5 * np.eye(n))
        return L, ell, sig

    def _predict(self, Xs, L, ell, sig):
        Xs = np.atleast_2d(Xs)
        n = L.shape[0]
        d = Xs.shape[1]
        v = np.linalg.solve(L, np.linalg.solve(L.T, Xs))
        Ks = sig * sig * np.exp(-0.5 * ((Xs[:, :, None] - Xs[:, None, :]) ** 2).sum(1) / ell ** 2)
        Kxs = sig * sig * np.exp(-0.5 * ((Xs[None, :, :] - Xs[:, None, :]) ** 2).sum(2) / ell ** 2)
        mean = Kxs @ v
        var = sig * sig - (Kxs * (Kxs @ v)).sum(1)
        return mean, np.maximum(var, 1e-12)

    def _ehvi(self, F, ref, mu, sd):
        F = np.asarray(F, dtype=float)
        ref = np.asarray(ref, dtype=float)
        n = F.shape[0]
        if n == 0:
            return np.zeros_like(mu[:, 0])
        F = F[(F < ref).all(axis=1)]
        if F.shape[0] == 0:
            return np.zeros_like(mu[:, 0])
        F = F[self._pareto(F)]
        if F.shape[0] == 0:
            return np.zeros_like(mu[:, 0])
        F = F[np.lexsort(F[:, ::-1])]
        hv = self._hypervolume(F, ref)
        m = mu.shape[0]
        out = np.zeros(m)
        for j in range(m):
            if mu[j, 0] >= ref[0] or mu[j, 1] >= ref[1]:
                out[j] = 0.0
                continue
            s = (mu[j, 0] - F[:, 0]) / sd[j, 0]
            p0 = norm.cdf(s)
            s1 = (mu[j, 1] - F[:, 1]) / sd[j, 1]
            p1 = norm.cdf(s1)
            w = (ref[0] - mu[j, 0]) * (ref[1] - mu[j, 1])
            gain = np.clip(w * (p0 * p1 + (1 - p0) * (1 - p1)) - hv, 0.0, None)
            out[j] = gain
        return out

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        ref = np.array([3000.0, 0.06])
        budget = self.budget

        def time_ok():
            return time.time() < deadline and len(self._X) < budget

        rng = np.random.default_rng(0)

        # LHS initialization
        def lhs_sample(n):
            idx = np.array([np.arange(n)[::-1]] * d)
            for i in range(d):
                idx[i] = np.random.permutation(n)
            return (idx + np.random.random((d, n))) / n

        self._X = []
        self._F = []

        n_init = min(budget, 24)
        X0 = lo + (hi - lo) * lhs_sample(n_init).T
        for x in X0:
            if not time_ok():
                break
            self._X.append(x)
            self._F.append(np.asarray(problem(x), dtype=float))

        if len(self._X) == 0:
            x = lo + 0.5 * (hi - lo)
            self._X.append(x)
            self._F.append(np.asarray(problem(x), dtype=float))

        self._X = np.array(self._X)
        self._F = np.array(self._F)

        # Surrogate loop
        while time_ok():
            n = len(self._X)
            L, ell, sig = self._gp(self._X, self._F, n)

            # Candidate pool: LHS + current points
            n_cand = min(budget - n, 40)
            cands = lo + (hi - lo) * lhs_sample(max(n_cand, 1)).T
            cands = np.vstack([cands, self._X])
            cands = np.unique(np.round(cands, 6), axis=0)

            mu, sd = self._predict(cands, L, ell, sig)
            sd = np.maximum(sd, 1e-6)
            acq = self._ehvi(self._F, ref, mu, sd)

            # Local polish around the top acquisition points
            top = np.argsort(acq)[::-1][:3]
            extra = []
            for t in top:
                c = cands[t]
                for _ in range(6):
                    step = 0.03 * (hi - lo)
                    nc = c + step * rng.standard_normal(d)
                    nc = np.clip(nc, lo, hi)
                    extra.append(nc)
            if extra:
                extra = np.array(extra)
                mu_e, sd_e = self._predict(extra, L, ell, sig)
                acq_e = self._ehvi(self._F, ref, mu_e, np.maximum(sd_e, 1e-6))
                cands = np.vstack([cands, extra])
                acq = np.concatenate([acq, acq_e])

            best = int(np.argmax(acq))
            x = cands[best]
            fx = np.asarray(problem(x), dtype=float)
            self._X = np.vstack([self._X, x])
            self._F = np.vstack([self._F, fx])

        # Final high-precision polish pass on the Pareto points
        if time.time() < deadline - 20 and len(self._X) > 0:
            keep = self._pareto(self._F)
            pf = self._X[keep]
            for x in pf:
                if time.time() >= deadline - 20:
                    break
                try:
                    res = minimize(lambda v: float(np.sum(np.asarray(problem(v), dtype=float) ** 2)),
                                   x, method='L-BFGS-B',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 30, 'maxfun': 200})
                    if res.success and np.isfinite(res.fun):
                        self._X = np.vstack([self._X, res.x])
                        self._F = np.vstack([self._F, np.asarray(problem(res.x), dtype=float)])
                except Exception:
                    pass

        return self._X
# EVOLVE-BLOCK-END
