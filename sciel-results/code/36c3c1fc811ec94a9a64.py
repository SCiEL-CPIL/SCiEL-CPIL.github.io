"""CMA-ES-style continuous optimizer for the BBOB f13 sharp-ridge task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with a deadline guard and a light final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0  # safety margin well under the 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # ---- Phase 0: cheap coarse scan to seed a decent starting point ----
        # f13 is a sharp ridge; a few well-spread probes help locate the basin.
        candidates = []
        base = np.zeros(dim)
        for ax in range(dim):
            for v in (-3.0, 0.0, 3.0):
                p = base.copy()
                p[ax] = v
                candidates.append(p)
        # plus a few random interior points
        for _ in range(8):
            candidates.append(rng.uniform(-4, 4, dim))
        # dedupe-ish: just evaluate all (cheap, 16+8=24 calls)
        best_x = None
        best_f = np.inf
        for p in candidates:
            if time.time() > deadline:
                break
            fp = func(np.clip(p, -5, 5))
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # ---- Phase 1: CMA-ES ----
        x = best_x.copy()
        f = best_f
        pop = max(4, int(round(4 + 3 * np.log(dim))))
        sigma = 2.0
        weights = np.exp(-np.arange(pop) / (pop / 2.0))
        weights /= weights.sum()
        mu = int(np.floor(pop / 2))
        mu = max(mu, 1)
        wmu = weights[:mu]
        wmu = wmu / wmu.sum()
        c_sigma = (mu + 2) / (pop + 4)
        c_c = (4 + mu / pop) / (pop + 4 + 2 * mu / pop)
        c_1 = (2 + 2.0 / (pop + 1.3) ** 2) / ((pop + 1.3) ** 2 + 2 + mu / pop)
        c_mu = min(1 - c_1,
                   2 * (sqrtmu - 1) / ((pop - 2 + sqrtmu) * (pop + 2 - 2 / sqrtmu)))
        sqrtmu = np.sqrt(np.sum(wmu ** 2))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        calls = 0

        def elapsed_ok():
            return time.time() < deadline

        # helper for CMA-ES update (inlined per generation for speed)
        while calls < budget and elapsed_ok():
            # generate population
            ys = []
            for _ in range(pop):
                B = np.linalg.cholesky(C)
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ z)
                ys.append(np.clip(y, -5, 5))
            calls += pop
            if not elapsed_ok():
                break
            fvals = [func(y) for y in ys]
            order = np.argsort(fvals)
            # update best
            if fvals[order[0]] < f:
                f = fvals[order[0]]
                x = ys[order[0]].copy()
            # recombination
            xnew = np.zeros(dim)
            for i in range(mu):
                xnew += wmu[i] * ys[order[i]]
            # step-size control
            damps = 1 + 2 * max(0, 1 / (sqrtmu * np.sqrt(dim)) - 1) + c_sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu) * (B @ (np.linalg.solve(B, (xnew - x) / sigma)))
            if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma:
                sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps) / chi_n - 1))
            # update mean and evolution path
            x = xnew
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu) * (xnew - x) / sigma
            # covariance update
            delta_h = (1.7 + 2 / (dim + 1)) * c_c
            hsig = 1 if (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (calls // pop + 1)))
                         < (delta_h + 1) * chi_n) else 0
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                yi = ys[order[i]] - xnew
                rankmu += wmu[i] * np.outer(yi, yi)
            C = (1 + c_1 * hsig * (dim + 1 - 2 / (dim + 1)) - c_1 - c_mu) * C \
                + c_1 * (rank1 + (1 - hsig) * (C - np.eye(dim))) \
                + c_mu * rankmu
            # symmetrize and ensure positive-definite-ish
            C = 0.5 * (C + C.T)
            # clamp sigma
            sigma = float(np.clip(sigma, 1e-8, 50.0))
            if not elapsed_ok():
                break

        # ---- Phase 2: light (1+1)-ES polish around the best point ----
        sigma = max(0.02, 0.05 * sigma)
        remaining = budget - calls
        for _ in range(remaining):
            if not elapsed_ok():
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < f:
                x, f = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85
            if sigma < 1e-10:
                break

        return x, f
# EVOLVE-BLOCK-END
