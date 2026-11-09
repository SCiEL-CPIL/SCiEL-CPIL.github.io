import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 4
        mu = 2
        w = np.array([1.0, 0.5])
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)

        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))

        # Multi-start
        starts = [np.zeros(dim)] + [rng.uniform(-4, 4, dim) for _ in range(3)]
        best_x, best_f = None, np.inf
        n_evals = 0
        for s in starts:
            s = np.clip(s, -5, 5)
            f = func(s)
            n_evals += 1
            if f < best_f:
                best_f, best_x = f, s.copy()

        x = best_x.copy()
        sigma = 1.5
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        gen = 0

        while n_evals < budget:
            if time.time() > deadline:
                break

            B = np.linalg.cholesky(C)
            off = [x + sigma * (B @ rng.standard_normal(dim)) for _ in range(lam)]

            off_f = []
            for o in off:
                oc = np.clip(o, -5, 5)
                f = func(oc)
                n_evals += 1
                off_f.append(f)
                if f < best_f:
                    best_f, best_x = f, oc.copy()

            if n_evals >= budget:
                break

            idx = np.argsort(off_f)
            so = [off[i] for i in idx]

            x_old = x.copy()
            x = sum(w[i] * so[i] for i in range(mu))

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B @ (x - x_old))
            gen += 1
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * gen))) < (1.4 + 2.0 / (dim + 1)) * chiN

            sigma *= np.exp(cs / (dim + 1) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(max(sigma, 1e-12), 1e3)

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) / sigma * (x - x_old)

            C = (1 + c1 * (1 - hsig) * cc * (2 - cc) - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                C += cmu * w[i] * np.outer(so[i] - x_old, so[i] - x_old)

            C = (C + C.T) / 2
            x = np.clip(x, -5, 5)

        return best_x, best_f
