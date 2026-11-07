"""CMA-ES black-box optimizer for the BBOB-style sharp-ridge f13."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


class Optimizer:
    """CMA-ES with a small ridge-detection pre-phase and a wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0          # safety margin below the 330 s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        calls = 0

        def ev(x):
            nonlocal calls
            calls += 1
            try:
                v = func(np.asarray(x, dtype=float))
            except Exception:
                return float("inf")
            if not np.isfinite(v):
                return float("inf")
            return float(v)

        # ---------- initial point (center is a safe prior for the box) ----------
        x0 = np.zeros(dim)
        f0 = ev(x0)
        best_x = x0.copy()
        best_f = f0

        # ---------- cheap ridge anisotropy probe ----------
        # A direction where a small step RAISES f is a steep wall -> ridge axis.
        probe = 0.35
        aniso = 0.0
        best_axis = None
        for _ in range(dim):
            d = rng.standard_normal(dim)
            d /= (np.linalg.norm(d) + 1e-12)
            fa = ev(x0 + probe * d)
            fb = ev(x0 - probe * d)
            spread = abs(fa - fb)
            if spread > aniso:
                aniso = spread
                best_axis = d
        if best_axis is not None:
            # seed covariance: strong variance along the ridge, small elsewhere
            lam = 25.0
            C = (lam - 1.0) * np.outer(best_axis, best_axis) + (1.0 / dim) * (
                np.eye(dim) - np.outer(best_axis, best_axis)
            )
        else:
            C = np.eye(dim)

        # ---------- CMA-ES ----------
        sigma = 1.5
        best_sigma = sigma
        best_C = C.copy()
        N = max(4, int(round(4 + 3 * math.log(dim))))
        mu = N // 2
        # recombination weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0.0, (mueff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 1) + 2 * w[0]) / ((dim + 2) ** 2 + mueff)
        c_mu = min(1 - c_1,
                   2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        while calls < budget:
            if time.time() > deadline - 3.0:
                break
            # sample
            ys, fs = [], []
            for _ in range(N):
                z = rng.standard_normal(dim)
                y = best_x + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5.0, 5.0)
                ys.append(y)
                fs.append(ev(y))
                if time.time() > deadline - 1.0:
                    break
            fs = np.array(fs, dtype=float)
            order = np.argsort(fs)
            fs = fs[order]
            ys = [ys[i] for i in order]

            # update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                best_sigma = sigma
                best_C = C.copy()

            # recombination
            ymean = np.zeros(dim)
            for i in range(mu):
                ymean += w[i] * ys[i]

            # step-size adaptation (path)
            ps = (1 - c_sigma) * ps + math.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                C_inv_sqrt @ (ymean - best_x) / sigma
            )
            hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1)))) \
                   < (1.4 + 2 / (dim + 1)) * chiN
            sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(max(sigma, 1e-12), 50.0)

            # covariance adaptation
            v = ymean - best_x
            pc = (1 - c_c) * pc + hsig * math.sqrt(c_c * (2 - c_c) * mueff) * v
            vC = v / sigma
            C = (1 + c_1 * (hsig - 1) + c_mu) * C \
                + c_1 * (pc @ pc.T - (1 - c_c) * (2 - c_c) * C) \
                + c_mu * sum(w[i] * np.outer(ys[i] - best_x, ys[i] - best_x)
                             for i in range(mu)) / (sigma ** 2)
            # symmetrize + repair
            C = 0.5 * (C + C.T)
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                # fall back to the last good state
                C = best_C.copy()
                sigma = best_sigma
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))

            best_x = ymean
            # keep best_x inside the box for the next iteration's centering
            best_x = np.clip(best_x, -5.0, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
