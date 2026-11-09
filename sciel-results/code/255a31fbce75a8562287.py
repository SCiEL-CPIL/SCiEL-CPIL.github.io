"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation) with a cheap initial bracketing
    stage to locate the global basin, exploiting the anisotropic geometry of
    the rotated Griewank-Rosenbrock landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start = time.time()
        DEADLINE = 320.0  # safety margin before the 330s kill

        rng = np.random.default_rng(12345)

        # ---------- Stage 1: cheap bracketing to find a good seed ----------
        candidates = [
            np.zeros(n),
            np.full(n, 2.0),
            np.full(n, -2.0),
            np.array([3.0, -3.0, 3.0, -3.0, 3.0])[:n],
            np.array([-3.0, 3.0, -3.0, 3.0, -3.0])[:n],
            np.full(n, 1.0),
            np.full(n, -1.0),
            np.array([4.0, 0.0, -4.0, 0.0, 4.0])[:n],
            np.array([0.0, 4.0, 0.0, -4.0, 0.0])[:n],
        ]
        for _ in range(8):
            candidates.append(rng.uniform(-4.5, 4.5, n))

        best_x = None
        best_f = np.inf
        fvals = []
        for p in candidates:
            if time.time() > DEADLINE:
                break
            p = np.clip(p, -5.0, 5.0)
            fp = float(func(p))
            fvals.append(fp)
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        fvals = np.array(fvals)
        if len(fvals) >= 2:
            spread = float(np.std(fvals))
            sigma0 = 0.3 if spread < 1e-3 else 0.8
        else:
            sigma0 = 1.0

        # ---------- Stage 2: CMA-ES from the best bracketing point ----------
        lam = int(4 + 3 * np.log(n)) if n > 1 else 2
        mu = lam // 2
        w = np.log(np.arange(mu, 0, -1)) - np.log(mu)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) /
                  ((n + 2) ** 2 + mu_eff))
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        mean = best_x.copy()
        C = np.eye(n)
        Cinv = np.eye(n)
        ps = np.zeros(n)
        ph = np.zeros(n)
        sigma = sigma0

        calls = len(candidates)

        def sqrtm(M):
            w_, v = np.linalg.eigh(M)
            w_ = np.clip(w_, 1e-12, None)
            return v @ (np.sqrt(w_)[:, None] * v.T)

        B = sqrtm(C)
        D = np.sqrt(np.clip(np.diag(C), 1e-12, None))

        while calls < budget:
            if time.time() > DEADLINE:
                break

            # Generate lambda offspring.
            ys = []
            xs = []
            for _ in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(n)
                y = sigma * (B @ (D * z))
                x = np.clip(mean + y, -5.0, 5.0)
                ys.append(y)
                xs.append(x)
                calls += 1

            # Evaluate.
            fxs = []
            for x in xs:
                if time.time() > DEADLINE:
                    fxs.append(np.inf)
                    continue
                fxs.append(float(func(x)))

            # Sort by fitness.
            order = np.argsort(fxs)
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            fxs = np.array([fxs[i] for i in order])

            # Update mean with top mu.
            new_mean = np.sum(w[:, None] * np.array(xs[:mu]), axis=0)
            new_mean = np.clip(new_mean, -5.0, 5.0)

            # Update covariance (standard CMA-ES form).
            C = (1 - c1 - cmu) * C
            for i in range(mu):
                d = np.array(xs[i]) - mean
                C += (cmu * w[i]) * np.outer(d, d)
            C = 0.5 * (C + C.T)

            # Evolution paths.
            ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) * (Cinv @ (new_mean - mean)) / sigma
            ph = (1 - cs) * ph + cs * np.sqrt(mu_eff) * (new_mean - mean) / sigma

            # Step size adaptation (1/5 rule on ps).
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-8, 50.0)

            # Update Cinv and B, D.
            try:
                Cinv = np.linalg.inv(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                C = C + 1e-8 * np.eye(n)
                Cinv = np.linalg.inv(C)
            B = sqrtm(C)
            D = np.sqrt(np.clip(np.diag(C), 1e-12, None))

            mean = new_mean
            if fxs[0] < best_f:
                best_f = float(fxs[0])
                best_x = np.array(xs[0]).copy()

            # Convergence check.
            if sigma < 1e-10 or np.all(np.abs(D) < 1e-10):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
