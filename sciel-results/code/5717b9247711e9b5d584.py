"""Two-phase optimizer: CMA-ES then local refinement for ridge functions."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        calls = [0]

        def counted(x):
            calls[0] += 1
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        # Phase 1: CMA-ES
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        lam = 4 + int(3 * np.log(max(dim, 2)))
        mu = lam // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x.copy()
        best_f = counted(best_x)

        phase1_max = int(budget * 0.55)
        for gen in range(phase1_max):
            if calls[0] >= budget - 15 or time.time() > deadline - 10:
                break

            L = np.linalg.cholesky(C)
            xs = np.zeros((lam, dim))
            fs = np.zeros(lam)

            for i in range(lam):
                if calls[0] >= budget - 5:
                    break
                z = rng.standard_normal(dim)
                xs[i] = x + sigma * (L @ z)
                fs[i] = counted(xs[i])
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = xs[i].copy()

            if calls[0] >= budget - 5:
                break

            order = np.argsort(fs)
            fs_s = fs[order]
            xs_s = xs[order]

            x_old = x.copy()
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * xs_s[i]

            pc = (1 - cc) * pc + cc / sigma * (x_new - x_old)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (L.T @ np.linalg.solve(L.T, (x_new - x_old)))

            if np.dot(ps, ps) / (1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / mu) * chi ** 2:
                rank_one = np.outer(ps, ps)
            else:
                rank_one = (1 + 2 / mu) * chi ** 2 * np.eye(dim)

            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                v = (xs_s[i] - x_old) / sigma
                rank_mu += w[i] * np.outer(v, v)

            C = (1 + c1 * cmu - c1) * C + c1 * (np.outer(x_new - x_old, x_new - x_old) / sigma ** 2 - np.eye(dim) + rank_one) + c1 * cmu * rank_mu
            C = (C + C.T) / 2

            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(dim)

            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            x = x_new

        # Phase 2: Local refinement with adaptive steps
        if calls[0] < budget - 5 and time.time() < deadline - 5:
            # Multi-start local search from best_x
            current_x = best_x.copy()
            current_f = best_f

            # Try coordinate-wise refinement
            for _ in range(3):
                if calls[0] >= budget - 3 or time.time() > deadline - 3:
                    break
                improved = False
                for coord in range(dim):
                    if calls[0] >= budget - 2:
                        break
                    for step in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
                        if calls[0] >= budget - 1:
                            break
                        for direction in [-1, 1]:
                            if calls[0] >= budget - 1:
                                break
                            trial = current_x.copy()
                            trial[coord] += direction * step
                            trial = np.clip(trial, -5, 5)
                            f_trial = counted(trial)
                            if f_trial < current_f:
                                current_f = f_trial
                                current_x = trial.copy()
                                improved = True
                                if f_trial < best_f:
                                    best_f = f_trial
                                    best_x = trial.copy()
                if not improved:
                    break

        return best_x, best_f
