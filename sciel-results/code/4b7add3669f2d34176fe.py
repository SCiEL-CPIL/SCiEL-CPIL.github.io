"""Rewritten optimizer: multi-restart CMA-ES with time guard for BBOB f4."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal call_count
            if call_count >= budget:
                return None
            fx = func(x)
            call_count += 1
            return fx

        def cma_es(x0, sigma0, max_calls, deadline):
            nonlocal best_f, best_x, call_count
            dim_local = dim
            lam = max(4, int(np.ceil(4 + 3 * np.log(dim_local))))
            if lam > max_calls // 2:
                lam = max(4, max_calls // 10)
            mu = lam // 2
            weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            cc = (4 + dim_local / (dim_local + 4)) / ((dim_local + 4) + 4 + dim_local / (dim_local + 4))
            cs = (dim_local + 1.7) / (dim_local + 9 + 2 * dim_local / (dim_local + 4))
            c1 = (2 - mu_eff / (dim_local + 1.8) + cc) / ((dim_local + 1.3) ** 2 + 1)
            cmu = (2 * (mu_eff / 2 - 1) - c1) / ((dim_local + 1.3) ** 2 - 3)

            x = x0.copy()
            sigma = sigma0
            C = np.eye(dim_local)
            B = np.eye(dim_local)
            D = np.ones(dim_local)
            p_c = np.zeros(dim_local)
            p_sigma = np.zeros(dim_local)

            fx = evaluate(x)
            if fx is None:
                return
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            gen = 0
            while call_count < budget and time.time() < deadline:
                if call_count >= budget or time.time() > deadline - 2.0:
                    break
                if call_count + lam > budget:
                    lam_remaining = budget - call_count
                    if lam_remaining < mu:
                        break
                    lam = lam_remaining

                ys = []
                fs = []
                for i in range(lam):
                    if call_count >= budget or time.time() > deadline - 1.0:
                        break
                    z = rng.standard_normal(dim_local)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = evaluate(y)
                    if fy is not None:
                        ys.append(y)
                        fs.append(fy)

                if len(ys) < mu:
                    break

                idx = np.argsort(fs)
                ys_sel = [ys[i] for i in idx[:mu]]
                fs_sel = [fs[i] for i in idx[:mu]]

                for i in range(mu):
                    if fs_sel[i] < best_f:
                        best_f = fs_sel[i]
                        best_x = ys_sel[i].copy()

                x_new = np.sum([weights[i] * ys_sel[i] for i in range(mu)], axis=0)

                # Update p_sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (x_new - x)))
                chi_n = np.sqrt(dim_local) * (1 - 1.0 / (4 * dim_local) + 1.0 / (21 * dim_local ** 2))
                hsig = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim_local + 1)) * chi_n else 0

                # Update p_c
                p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new - x) / sigma

                # Update C
                C = (1 - c1 - cmu) * (B @ (D ** 2) @ B.T)
                C += c1 * (np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * (B @ (D ** 2) @ B.T))
                for i in range(mu):
                    y_i = (ys_sel[i] - x) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)

                # Update B, D
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    B = eigvecs
                    D = np.sqrt(eigvals)
                except np.linalg.LinAlgError:
                    break

                x = x_new
                sigma *= np.exp(cs / (2 * dim_local) * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                if sigma < 1e-10:
                    break

                gen += 1

        # Multi-restart strategy
        restarts = 4
        calls_per = budget // restarts

        for r in range(restarts):
            if call_count >= budget or time.time() > deadline - 5.0:
                break
            if r == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.5
            else:
                x0 = rng.uniform(-3, 3, dim)
                sigma0 = 2.0

            remaining = budget - call_count
            cma_es(x0, sigma0, remaining, deadline)

        return best_x, best_f
