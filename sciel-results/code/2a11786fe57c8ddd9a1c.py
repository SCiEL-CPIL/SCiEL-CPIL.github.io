"""Fresh design: population CMA-ES with a wall-clock deadline guard for BBOB f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Population CMA-ES (black-box) with a hard wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin well inside the 300 s limit

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(0)

        # Population size and strategy parameters
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # ~10 for dim=5
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / (w @ w)
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = (2.0 / ((dim + 1.3) ** 2 + mueff)) / (dim + 4.0)
        cmu = min(1.0 - c1, 2.0 * ((mueff + 2.0) / (dim + 2.0) ** 2 - 1.0) / (dim + 4.0))
        chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

        # Initialisation: sample a few starts, keep the best as the mean
        x0 = np.zeros(dim)
        best_f = np.inf
        for _ in range(8):
            cand = rng.uniform(-4.0, 4.0, dim)
            f = func(cand)
            if f < best_f:
                best_f = f
                x0 = cand.copy()

        xmean = x0.copy()
        fbest = best_f
        fmean = best_f

        # Start with a moderately large sigma to cover the box
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        call_count = 8

        while call_count < budget:
            if time_left() < 0.5:
                break

            # Generate population
            ys = np.empty((lam, dim))
            xs = np.empty((lam, dim))
            fs = np.empty(lam)
            for i in range(lam):
                if time_left() < 0.5:
                    break
                yi = B @ (D * rng.standard_normal(dim))
                ys[i] = yi
                xi = np.clip(xmean + sigma * yi, -5.0, 5.0)
                xs[i] = xi
                fs[i] = func(xi)
                call_count += 1
            lam_used = call_count - (call_count - lam)  # keep it simple: use full lam
            lam_used = min(lam, call_count)

            # Track global best
            i_best_local = int(np.argmin(fs))
            if fs[i_best_local] < fbest:
                fbest = fs[i_best_local]
                xmean_best = xs[i_best_local].copy()

            # Sort by fitness
            order = np.argsort(fs)
            fs_sorted = fs[order]
            xs_sorted = xs[order]
            ys_sorted = ys[order]

            # Update mean (weighted)
            w_use = w[:mu]
            xmean = xs_sorted[:mu].T @ w_use

            # Fitness-dependent weights for covariance
            w_cov = w.copy()
            w_cov = w_cov / w_cov.max()
            w_cov -= 0.2
            w_cov[w_cov < 0] = 0
            mueff_cov = (w_cov[:mu] @ w_cov[:mu]) if (w_cov[:mu] @ w_cov[:mu]) > 1e-12 else 1.0
            mueff_cov = max(mueff_cov, 1e-12)

            # CMA update
            ydiff = (xs_sorted[:mu] - xmean).T
            # pc update
            pc = (1.0 - cc) * pc + cc * (np.sqrt(mueff) * cs * chiN) * (
                (xs_sorted[:mu] - xmean).T @ w_use
            ) / sigma
            # ps update
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (
                B @ (D * (B.T @ (pc / sigma)))
            )
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (call_count) / dim))) < (
                1.4 + 2.0 / (dim + 1.0)
            ) * chiN
            # Covariance matrix update (use eigen-decomposition-free form via rank updates)
            # We maintain B and D; recompute from C for stability each step.
            C = (B * D) @ (B * D).T
            delta_hsig = 1.0 - hsig
            C = (1.0 + c1 * delta_hsig - c1 - cmu) * C + c1 * np.outer(pc, pc)
            for k in range(mu):
                ywk = (B @ (D * ys_sorted[k])) / sigma
                C += cmu * w_cov[k] * np.outer(ywk, ywk)

            # Eigen-decompose for B, D (stable)
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-20, None)
            D = np.sqrt(evals)
            B = evecs

            # Sigma update
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1.0))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Safety: keep mean inside the box
            xmean = np.clip(xmean, -5.0, 5.0)

        return xmean, fbest
# EVOLVE-BLOCK-END
