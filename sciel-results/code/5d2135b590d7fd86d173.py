"""CMA-ES with IPOP restarts for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with IPOP restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        dim = self.dim
        budget = self.budget

        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        def run_cma_es(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with sigma0 for up to max_evals evaluations."""
            nonlocal call_count, best_f, best_x, deadline, budget

            n = dim
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            csig = (mueff + 2) / (n + mueff + 5)
            dsig = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + csig
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((n + 2) ** 2 + mueff))

            xmean = x0.copy().astype(np.float64)
            sigma = sigma0
            C = np.eye(n)
            Cinv = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chn = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            Cchol = np.linalg.cholesky(C)

            evals = 0

            while evals < max_evals:
                if time.time() >= deadline or call_count >= budget:
                    break

                # Generate and evaluate population
                pop = []
                for i in range(lam):
                    if time.time() >= deadline or call_count >= budget:
                        break
                    y = rng.standard_normal(n)
                    z = Cchol @ y
                    xi = np.clip(xmean + sigma * z, -5.0, 5.0)
                    fi = wrapped_func(xi)
                    evals += 1
                    pop.append((fi, xi))

                if len(pop) < mu:
                    break

                pop.sort(key=lambda t: t[0])
                sel = [p[1] for p in pop[:mu]]
                sel_f = [p[0] for p in pop[:mu]]

                if sel_f[0] < best_f:
                    best_f = sel_f[0]
                    best_x = sel[0].copy()

                xnew = np.sum(w[:, None] * np.array(sel), axis=0)
                xold = xmean.copy()
                diff = xnew - xold

                # Step size update
                ps = (1 - csig) * ps + np.sqrt(csig * (2 - csig) * mueff) / chn * (Cinv @ diff) / sigma
                sigma *= np.exp(csig / dsig * (np.linalg.norm(ps) / chn - 1))
                sigma = np.clip(sigma, 1e-12, 1e4)

                xmean = xnew

                # Covariance update
                if np.linalg.norm(ps) < (1.4 + 1.0 / csig) * chn:
                    v = diff / sigma
                    pc = (1 - cc) * pc + cc * v
                    r1 = np.outer(pc, pc)
                else:
                    r1 = np.eye(n)

                rm = np.zeros((n, n))
                for i in range(mu):
                    vi = (sel[i] - xold) / sigma
                    rm += w[i] * np.outer(vi, vi)

                C = (1 + cc * cc * (1 - cc) + c1 + cmu - cc) * C + c1 * (r1 - C) + cmu * rm

                # Ensure PD
                ev, evec = np.linalg.eigh(C)
                ev = np.maximum(ev, 1e-12)
                C = evec @ np.diag(ev) @ evec.T
                Cinv = np.linalg.inv(C)
                Cchol = np.linalg.cholesky(C + 1e-10 * np.eye(n))

                if sigma < 1e-10 or np.max(ev) < 1e-12:
                    break

        # IPOP-CMA-ES loop
        restarts = 0
        max_restarts = 8
        sigma_base = 1.0

        while call_count < budget and restarts < max_restarts:
            if time.time() >= deadline:
                break

            remaining = budget - call_count
            lam = 4 + int(3 * np.log(dim))
            if remaining < lam:
                break

            if restarts == 0:
                x0 = np.zeros(dim)
                sigma0 = sigma_base
            else:
                x0 = best_x.copy() if best_x is not None else np.zeros(dim)
                sigma0 = sigma_base * (2.0 ** restarts)
                sigma0 = min(sigma0, 5.0)

            alloc = min(remaining, max(lam + 5, remaining // 2))
            if alloc < lam + 5:
                alloc = remaining

            run_cma_es(x0, sigma0, alloc)
            restarts += 1

            if best_f < -80:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
