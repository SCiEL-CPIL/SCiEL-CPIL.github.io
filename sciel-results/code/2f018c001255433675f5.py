"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize as _scipy_minimize


class Optimizer:
    """CMA-ES global search with local refinement for shifted/rotated
    Rosenbrock on [-5, 5]^dim.  Uses population-based adaptive search
    to navigate the landscape, then polishes with L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()

        # --- deadline guard -------------------------------------------------
        def deadline():
            return time.perf_counter() - t0 > 270.0

        # --- evaluation counter ---------------------------------------------
        eval_count = [0]
        best_f = [float("inf")]
        best_x = [None]

        def eval_func(x):
            if eval_count[0] >= budget or deadline():
                return best_f[0]
            eval_count[0] += 1
            try:
                f = float(func(np.asarray(x, dtype=float)))
            except Exception:
                return float("inf")
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = np.array(x, dtype=float)
            return f

        # --- CMA-ES implementation ------------------------------------------
        def cma_es(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with initial sigma. Returns (best_x, best_f)."""
            n = dim
            lam = max(4, int(4 + 3 * math.log(n)))
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / (w ** 2).sum()
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            x_mean = np.array(x0, dtype=float).copy()
            sigma = sigma0
            C = np.eye(n)
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_chol = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            local_best_f = float("inf")
            local_best_x = x_mean.copy()
            # Initial population
            pop = []
            for i in range(lam):
                z = np.random.standard_normal(n)
                x_i = x_mean + sigma * C_chol @ z
                f_i = eval_func(x_i)
                pop.append((f_i, x_i))
                if f_i < local_best_f:
                    local_best_f = f_i
                    local_best_x = x_i.copy()
                if eval_count[0] >= max_evals or deadline():
                    break

            while eval_count[0] < max_evals and not deadline():
                pop.sort(key=lambda t: t[0])
                selected = pop[:mu]
                x_new_mean = sum(w[i] * selected[i][1] for i in range(mu))
                try:
                    C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-12 * np.eye(n)))
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)
                diff = x_new_mean - x_mean
                p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ diff / sigma
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * math.sqrt(1 - (1 - c_sigma) ** 2) else 0.0
                p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mueff) * diff / sigma
                y = C_inv_sqrt @ diff / sigma
                C = (1 + c_1 * h_sigma * (2 - c_c) - c_1 - c_mu) * C + c_1 * (p_c @ p_c - C) + c_mu * sum(w[i] ** 2 * (selected[i][1] - x_mean) @ (selected[i][1] - x_mean).T / sigma ** 2 for i in range(mu))
                C = (C + C.T) / 2
                try:
                    C_chol = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                except np.linalg.LinAlgError:
                    C_chol = np.eye(n)
                sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / math.sqrt(2 * n) - 1))
                x_mean = x_new_mean
                pop = []
                for i in range(lam):
                    z = np.random.standard_normal(n)
                    x_i = x_mean + sigma * C_chol @ z
                    f_i = eval_func(x_i)
                    pop.append((f_i, x_i))
                    if f_i < local_best_f:
                        local_best_f = f_i
                        local_best_x = x_i.copy()
                    if eval_count[0] >= max_evals or deadline():
                        break
            return local_best_x, local_best_f

        # --- Local refinement ------------------------------------------------
        def local_refine(x0, remaining):
            bx = np.clip(x0.copy(), -5.0, 5.0)
            try:
                bf = float(func(bx))
            except Exception:
                return bx, float("inf")
            eval_count[0] += 1
            if bf < best_f[0]:
                best_f[0] = bf
                best_x[0] = bx.copy()

            if eval_count[0] < budget - 10 and not deadline():
                try:
                    res = _scipy_minimize(
                        eval_func,
                        bx,
                        method="L-BFGS-B",
                        bounds=[(-5.0, 5.0)] * dim,
                        options={"maxiter": 50, "maxfun": 40, "gtol": 1e-8},
                    )
                    rx = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
                    rf = eval_func(rx)
                except Exception:
                    pass

            if eval_count[0] < budget - 5 and not deadline():
                try:
                    res = _scipy_minimize(
                        eval_func,
                        best_x[0] if best_x[0] is not None else bx,
                        method="Nelder-Mead",
                        bounds=[(-5.0, 5.0)] * dim,
                        options={"maxiter": 30, "maxfev": 25, "xatol": 1e-6},
                    )
                    rx = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
                    rf = eval_func(rx)
                except Exception:
                    pass

            return best_x[0] if best_x[0] is not None else bx, best_f[0]

        # --- Main search loop ------------------------------------------------
        rng = np.random.default_rng(42)

        # Phase 1: CMA-ES with moderate initial sigma
        x0 = np.zeros(dim)
        sigma0 = 2.5
        max_cma_evals = min(350, budget - 50)
        cx, cf = cma_es(x0, sigma0, max_cma_evals)
        if cf < best_f[0]:
            best_f[0] = cf
            best_x[0] = cx

        # Phase 2: Second CMA-ES if budget remains
        if eval_count[0] < budget - 40 and not deadline():
            x0_2 = rng.uniform(-3.0, 3.0, dim)
            sigma0_2 = 1.5
            max_cma2 = min(100, budget - eval_count[0] - 30)
            if max_cma2 > 20:
                cx2, cf2 = cma_es(x0_2, sigma0_2, max_cma2)
                if cf2 < best_f[0]:
                    best_f[0] = cf2
                    best_x[0] = cx2

        # Phase 3: Local refinement
        if best_x[0] is not None and eval_count[0] < budget - 10 and not deadline():
            local_refine(best_x[0], budget - eval_count[0])

        # Fallback
        if best_x[0] is None:
            best_x[0] = np.zeros(dim)
            try:
                best_f[0] = float(func(best_x[0]))
            except Exception:
                best_f[0] = float("inf")

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
