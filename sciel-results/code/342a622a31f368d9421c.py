import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0
        n = self.dim
        budget = self.budget
        best_x = np.zeros(n)
        best_f = float('inf')
        used = 0

        def ev(x):
            nonlocal used, best_f, best_x
            f = func(np.clip(x, -5, 5))
            used += 1
            if f < best_f:
                best_f = f
                best_x = np.clip(x, -5, 5).copy()
            return f

        # Phase 1: Multi-start local search
        for s in range(6):
            if used >= budget or time.time() > deadline - 30:
                break
            x0 = np.random.uniform(-4, 4, n)
            sig = 2.5
            for g in range(12):
                if used >= budget or time.time() > deadline - 30:
                    break
                lam = 6
                X = x0[None, :] + sig * np.random.randn(lam, n)
                for i in range(lam):
                    if used >= budget or time.time() > deadline - 30:
                        break
                    ev(X[i])
                x0 = 0.6 * x0 + 0.4 * best_x
                sig *= 0.85

        # Phase 2: CMA-ES refinement
        if used < budget and time.time() < deadline - 50:
            self._cmaes(ev, best_x, n, budget - used, deadline)

        return best_x, best_f

    def _cmaes(self, ev, x0, n, rem, deadline):
        if rem < 12:
            return
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        me = 1.0 / (w @ w)
        cs = (me + 2) / (n + me + 5)
        ds = 1 + 2 * max(0, (me - 1) / (n + 1)) + cs
        cc = (4 + me / n) / (n + 4 + 2 * me / n)
        c1 = 2 / ((n + 1.3)**2 + me)
        cm = min(1 - c1, 2 * (me - 2 + 1 / me) / ((n + 2)**2 + me))
        xm = x0.copy()
        sig = 1.0
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        En = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        left = rem
        while left >= lam and time.time() < deadline - 15:
            try:
                B = np.linalg.cholesky(C)
            except:
                C, B, sig, ps, pc = np.eye(n), np.eye(n), 1.0, np.zeros(n), np.zeros(n)
            X = xm[None, :] + sig * (np.random.randn(lam, n) @ B.T)
            F = np.array([ev(X[i]) for i in range(lam)])
            left -= lam
            idx = np.argsort(F)
            Xs = X[idx]
            xn = (w[:, None] * Xs[:mu]).sum(axis=0)
            try:
                Cinv = np.linalg.inv(B) @ np.linalg.inv(B.T)
            except:
                Cinv = np.eye(n)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * me) * (Cinv @ ((xn - xm) / sig))
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * me) * ((xn - xm) / sig)
            ys = (Xs[:mu] - xm[None, :]) / sig
            C = (1 - c1 - cm) * C + c1 * np.outer(pc, pc) + cm * np.sum(w[:, None, None] * ys[:, :, None] * ys[:, None, :], axis=0)
            C = (C + C.T) / 2
            xm = xn
            sig = sig * np.exp(cs / ds * (np.linalg.norm(ps) / En - 1))
            sig = np.clip(sig, 1e-10, 1e5)
            if np.any(np.linalg.eigvalsh(C) < 1e-10):
                C, sig, ps, pc = np.eye(n), 1.0, np.zeros(n), np.zeros(n)
