"""CMA-ES based optimizer for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_time():
            nonlocal calls_used
            return time.time() < deadline and calls_used < budget

        def cma_es(x0, sigma0, max_iter, func):
            nonlocal calls_used, best_x, best_f
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = int(1 / np.sum(w**2))
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            mean = np.array(x0, dtype=float).copy()
            sigma = sigma0
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            ps = np.zeros(n)
            hsig = np.sqrt((1 - cc) / (1 + cc)) * (n + 1) * cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            w_mu = w[:mueff]

            for it in range(max_iter):
                if not check_time():
                    return
                old_B_D = B @ D
                z = np.random.randn(lam, n)
                y = mean + sigma * (z @ old_B_D.T)
                y = np.clip(y, -5, 5)
                fvals = np.empty(lam)
                for i in range(lam):
                    fvals[i] = func(y[i])
                    calls_used += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = y[i].copy()
                    if not check_time():
                        return
                if calls_used >= budget:
                    return
                idx = np.argsort(fvals)
                y_sorted = y[idx]
                fvals_sorted = fvals[idx]
                mean_new = mean + sigma * np.dot(w_mu, y_sorted[:mueff])
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B.T @ (D * (mean_new - mean)))
                hsig_val = np.sqrt((1 - cc) / (1 + cc)) * (n + 1) * cs
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (it + 1))) < (1.4 + 2 / (n + 1)) * hsig_val:
                    c = c1 + cmu
                else:
                    c = c1
                y_diff = (y_sorted[:mueff] - mean) / sigma
                C = (1 - c1 - cmu + c) * C + \
                    c1 * np.outer(mean_new - mean, mean_new - mean) / sigma**2 + \
                    cmu * (y_diff @ y_diff.T * w_mu[:, None]).sum(axis=0)
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)
                mean = mean_new
                if sigma < 1e-12:
                    return

        # Restart strategy
        num_restarts = 4
        iters_per_restart = max(10, budget // (num_restarts * lam))

        for r in range(num_restarts):
            if not check_time():
                break
            if r == 0:
                x0 = np.zeros(dim) + np.random.randn(dim) * 1.5
                sigma0 = 2.5
            else:
                if best_x is not None:
                    x0 = best_x + np.random.randn(dim) * 1.5
                else:
                    x0 = np.random.uniform(-5, 5, dim)
                sigma0 = 3.0

            x0 = np.clip(x0, -5, 5)
            cma_es(x0, sigma0, iters_per_restart, func)

        return best_x, best_f
# EVOLVE-BLOCK-END
