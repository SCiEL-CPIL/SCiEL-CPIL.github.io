"""Multi-start CMA-ES with time guard and polish for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, rng, x0, sigma0, budget, deadline, best_x, best_f,
               min_f=None):
        """Run one CMA-ES instance. Returns (best_x, best_f, calls_used)."""
        n = self.dim
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / n) / ((n + 4) + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (2 - cc + cs)
        cmu = (2 * (mueff / mueff - 1) * (0.5 + cs / cc)) / (
            (n + 2) ** 2 + mueff)
        cmu = min(cmu, c1 * (3 - cc))

        sigma = sigma0
        xmean = np.array(x0, dtype=float)
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        # Cholesky of C
        try:
            B = np.linalg.cholesky(C)
            D = np.sqrt(np.diag(B @ B.T))
        except Exception:
            B = np.eye(n)
            D = np.ones(n)
        invsqrt = np.ones(n)
        Dinv = 1.0 / D
        chiN = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        best_x = np.array(best_x, dtype=float)
        best_f = float(best_f)
        calls = 0

        while calls < budget:
            if time.time() > deadline:
                break
            # sample
            z = rng.standard_normal((lam, n))
            pops = xmean[None, :] + sigma * (z @ B.T)
            # clip
            pops = np.clip(pops, -5, 5)
            # evaluate
            vals = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                calls += 1
                if calls > self.budget:
                    break
                try:
                    vals[i] = float(func(pops[i]))
                except Exception:
                    vals[i] = np.inf
            if calls > self.budget:
                break
            # update best
            order = np.argsort(vals)
            for i in order:
                if vals[i] < best_f:
                    best_f = vals[i]
                    best_x = pops[i].copy()
            # selection
            idx = order[:mu]
            y = (pops[idx] - xmean[None, :]) / sigma
            yw = (w[:, None] * y).sum(axis=0)
            xnew = xmean + sigma * yw
            # step-size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (
                Dinv @ (B.T @ (yw / np.sqrt(sigma) if sigma > 0 else 0)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1) / (n + cs)))) < (
                (1 + 2 / (n + 1)) * chiN)
            pc = (1 - cc) * pc + hsig * cc * yw
            # covariance update
            yw_mat = yw[None, :]
            rank1 = np.outer(yw_mat[0], yw_mat[0])
            yw_sel = y / sigma if sigma > 0 else y
            rankmu = (w[:, None, None] *
                      np.einsum('ij,ik->ijk', yw_sel, yw_sel)).sum(axis=0)
            C = ((1 + c1 * cc * (cc - 1) - c1 - cmu) * C +
                 c1 * (np.outer(pc, pc) + (cc - cc * cc) * C) +
                 cmu * rankmu)
            # symmetrize
            C = 0.5 * (C + C.T)
            # recompute Cholesky
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(B @ B.T))
                Dinv = 1.0 / D
            except Exception:
                # add small diagonal and retry
                C = C + 1e-12 * np.eye(n)
                try:
                    B = np.linalg.cholesky(C)
                    D = np.sqrt(np.diag(B @ B.T))
                    Dinv = 1.0 / D
                except Exception:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    Dinv = np.ones(n)
            # update mean
            xmean = xnew
            # step-size update
            sigma = sigma * np.exp(cs / (5 * n) * (np.linalg.norm(ps) / chiN - 1))
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 1e3:
                sigma = 1e3
            # termination: tiny step
            if np.linalg.norm(yw) < 1e-10 and sigma < 1e-8:
                break
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        total_budget = self.budget
        n = self.dim
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Phase 1: multi-start CMA-ES with a few restarts
        # Use 3 starts: one near center, two random
        starts = []
        # center
        starts.append(np.zeros(n))
        starts.append(rng.uniform(-3, 3, n))
        starts.append(rng.uniform(-4.5, 4.5, n))

        # Budget allocation: give more to first start
        # First start gets 60% of budget, remaining split
        b1 = int(total_budget * 0.5)
        b2 = int(total_budget * 0.25)
        b3 = total_budget - b1 - b2
        budgets = [b1, b2, b3]

        for si, (x0, b) in enumerate(zip(starts, budgets)):
            if calls_used >= total_budget:
                break
            if time.time() > deadline:
                break
            sigma0 = 2.0
            if si == 0:
                sigma0 = 2.5
            bx, bf, c = self._cmaes(func, rng, x0, sigma0, b,
                                    deadline, best_x if best_x is not None else x0,
                                    best_f if np.isfinite(best_f) else np.inf)
            calls_used += c
            if bf < best_f:
                best_f = bf
                best_x = bx

        # Phase 2: final high-precision polish around best_x
        if best_x is not None and calls_used < total_budget and time.time() < deadline:
            remaining = total_budget - calls_used
            # Use a small CMA-ES with tiny sigma for polish
            sigma_polish = 1e-3
            # Try a few small sigma values
            for sp in [1e-2, 1e-3, 1e-4]:
                if calls_used >= total_budget:
                    break
                if time.time() > deadline:
                    break
                bx, bf, c = self._cmaes(func, rng, best_x, sp,
                                        min(remaining, 50),
                                        deadline, best_x, best_f)
                calls_used += c
                if bf < best_f:
                    best_f = bf
                    best_x = bx

        # Final local refinement: coordinate descent / small random jitter
        if best_x is not None and calls_used < total_budget and time.time() < deadline:
            remaining = total_budget - calls_used
            # Small random jitter refinement
            jitter_sigma = 1e-4
            for _ in range(min(remaining, 100)):
                if time.time() > deadline:
                    break
                if calls_used >= total_budget:
                    break
                cand = np.clip(best_x + jitter_sigma * rng.standard_normal(n), -5, 5)
                calls_used += 1
                try:
                    cf = float(func(cand))
                except Exception:
                    cf = np.inf
                if cf < best_f:
                    best_f = cf
                    best_x = cand.copy()
                    jitter_sigma *= 0.5
                else:
                    jitter_sigma *= 0.8
                if jitter_sigma < 1e-8:
                    break

        if best_x is None:
            best_x = np.zeros(n)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = np.inf
        return best_x, best_f
# EVOLVE-BLOCK-END
