"""Optimizer for BBOB-style f17 (Schaffers F7) on [-5,5]^5.
Strategy: CMA-ES for global search, then L-BFGS-B local polish.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class _Stop(Exception):
    pass


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin under 300s

        state = {"calls": 0, "best_x": None, "best_f": np.inf}

        def wrapper(x):
            state["calls"] += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            if fx < state["best_f"]:
                state["best_f"] = fx
                state["best_x"] = x.copy()
            return fx

        # --- Phase 1: CMA-ES (global) ---
        cma_budget = max(int(self.budget * 0.8), 60)
        self._cma_es(wrapper, np.zeros(self.dim), 2.0, cma_budget, deadline)

        # --- Phase 2: L-BFGS-B local polish ---
        if state["best_x"] is not None and state["calls"] < self.budget - 5 and time.time() < deadline:
            def lfgs_obj(x):
                if time.time() > deadline:
                    raise _Stop()
                return wrapper(x)

            bounds = [(-5.0, 5.0)] * self.dim
            remaining = self.budget - state["calls"] - 2
            for _ in range(2):
                if state["calls"] >= self.budget - 5 or time.time() > deadline:
                    break
                try:
                    minimize(
                        lfgs_obj,
                        state["best_x"].copy(),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": 150, "maxfun": max(remaining, 10)},
                    )
                except _Stop:
                    break
                except Exception:
                    break
                remaining = self.budget - state["calls"] - 2

        if state["calls"] == 0:
            wrapper(np.zeros(self.dim))

        return state["best_x"], state["best_f"]

    def _cma_es(self, func, x0, sigma0, budget, deadline):
        n = len(x0)
        pop = max(4 + int(3 * np.log(n)), 4)
        iters = max(1, budget // pop)

        m = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        cs = (n + 2) / (n + 5)
        ds = 1 + 2 * max(0.0, np.sqrt((n - 1) / (n + 3)) - 1.0) + cs
        c1 = 2.0 / ((n + 1.3) ** 2 + 10.0)
        cmu = min(0.5 - c1, 2.0 / ((n + 2) ** 2 + 5.0))
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
        ps = np.zeros(n)
        pc = np.zeros(n)

        weights = np.exp(-np.arange(1, pop + 1) / (0.35 * (n + pop) / 2.0))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        mu = int(np.floor(mu_eff))

        for it in range(iters):
            if time.time() > deadline:
                break
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C + 1e-14 * np.eye(n))
            except np.linalg.LinAlgError:
                C = (C + C.T) / 2.0
                L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
            # Generate population
            ys = [np.random.standard_normal(n) @ L.T for _ in range(pop)]
            xs = [m + sigma * y for y in ys]
            fvals = []
            for xi in xs:
                fvals.append(func(np.clip(xi, -5.0, 5.0)))
            order = np.argsort(fvals)
            ys_s = [ys[i] for i in order]
            xs_s = [xs[i] for i in order]

            # Update mean
            m_new = sum(weights[i] * xs_s[i] for i in range(mu))
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ (m_new - m))
            # Update pc
            norm_ps = np.linalg.norm(ps)
            denom = np.sqrt(1.0 - (1.0 - cs) ** (2 * (it + 1)))
            if norm_ps / denom < (1.4 + 1.0 / n) * chi_n:
                pc = (1 - cs) * pc + cs * (L.T @ (m_new - m))
            # Sigma update
            hsig = (norm_ps / np.sqrt(1.0 - (1.0 - cs) ** (it + 1))) / chi_n < (1.4 + 1.0 / n)
            c_sigma = (1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)) * cs
            sigma *= np.exp((c_sigma / ds) * (norm_ps / chi_n - 1.0))
            # Covariance update
            delta_hsig = (1.0 - hsig) * c1 * ds
            C = (1.0 + c1 * hsig * (2.0 - c1) - c1 - cmu + delta_hsig) * C
            C += c1 * np.outer(L.T @ (m_new - m), L.T @ (m_new - m))
            C += cmu * sum(
                weights[i] * np.outer(L.T @ (xs_s[i] - m), L.T @ (xs_s[i] - m))
                for i in range(mu)
            )
            C += delta_hsig * np.outer(pc, pc)
            C = (C + C.T) / 2.0
            m = m_new

        return m


# EVOLVE-BLOCK-END
