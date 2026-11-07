"""Rewritten optimizer: CMA-ES for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with deadline guard for f18 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 325.0  # safety margin before 330s kill
        rng = np.random.default_rng(12345)

        # --- CMA-ES parameters ---
        lam = max(4, int(4 + 3 * np.log(d)))
        w = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
        w = w / w.sum()
        mueff = int(np.floor(sum(w[: int(np.ceil(1 / w[0]))]) ** -1) if False else
                        np.floor(1.0 / np.sum(w[: int(np.ceil(1.0 / w[0]))] ** 2)))
        mueff = max(1, min(mueff, lam - 1))
        mu = int(np.floor(lam / 2))
        ws = w[:mu]
        ws = ws / ws.sum()
        mueff = int(np.floor(1.0 / np.sum(ws ** 2)))
        mueff = max(1, min(mueff, mu))

        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = (2 - cc + min(1 - cs, 1) * (2 - 1 / (d + 1.8) - cc)) / \
             ((d + 1.5) ** 2 + mueff)
        cmu = min(1 - c1,
                  2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Initial state
        xmean = np.zeros(d)
        sigma = 1.0
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        ps = np.zeros(d)
        pc = np.zeros(d)
        hsigma = (1 + 0.34) / (1 - cc)

        # Call budget tracking
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if x.shape != (d,):
                x = x.reshape(d)
            try:
                f = float(func(x))
            except Exception:
                # Exception: abort search, keep calls made
                return None
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation at center
        f0 = safe_call(xmean)
        if f0 is None:
            return (best_x, best_f) if best_x is not None else (np.zeros(d), np.inf)

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sample population
            ys = np.empty((lam, d))
            fs = np.empty(lam)
            for i in range(lam):
                # B * z / D
                z = rng.standard_normal(d)
                y = xmean + sigma * (B @ z) / D
                f = safe_call(y)
                if f is None:
                    # aborted
                    break
                ys[i] = y
                fs[i] = f
                if calls >= budget:
                    break

            if calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Update mean (weighted average of top mu)
            ymean = np.sum(ws[:, None] * ys[:mu], axis=0)

            # Update evolution paths
            C_sqrt_inv = B / D  # C^(-1/2)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * \
                 (C_sqrt_inv @ (ymean - xmean) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1)))) <= \
                   (1.4 + 2 / (d + 1)) * chi
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * \
                 (ymean - xmean) / sigma

            # Update covariance
            delta_hs = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(pc, pc)
            diff = (ys[:mu] - xmean) / sigma
            rank_mu = np.outer(diff.T @ ws, diff) / mueff
            C = (1 + c1 * delta_hs - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            # Update sigma (CSA)
            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Update mean
            xmean = ymean

            # Recompute eigendecomposition (C must be PSD)
            try:
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.clip(eigval, 1e-12, None)
                B = eigvec
                D = np.sqrt(eigval)
            except np.linalg.LinAlgError:
                C = np.eye(d)
                B = np.eye(d)
                D = np.ones(d)

        # Final: return best
        if best_x is None:
            best_x = xmean.copy()
            best_f = safe_call(best_x)
            if best_f is None:
                best_f = np.inf
        return best_x, best_f
# EVOLVE-BLOCK-END
