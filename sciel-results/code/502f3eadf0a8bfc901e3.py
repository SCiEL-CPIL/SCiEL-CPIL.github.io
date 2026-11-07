"""Rewritten program: CMA-ES for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with stagnation restart and wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma, budget, deadline, rng, state):
        """Run CMA-ES until budget or deadline. Returns (best_x, best_f)."""
        dim = self.dim
        # population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # step size adaptation
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (dim + 2)) + c_sigma
        # covariance adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (2 - c_c)
        c_mu = min(1 - c_1, (2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)) * (c_c + c_1))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        Csqrt = np.eye(dim)
        Cinv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        best_x = x0.copy()
        best_f = func(x0)
        state["calls"] += 1
        gen = 0
        no_improve = 0

        while state["calls"] < budget:
            if time.time() > deadline:
                break
            gen += 1
            # sample population
            z = rng.standard_normal((lam, dim))
            # apply inverse sqrt of C
            y = np.array([z[i] @ Cinv_sqrt for i in range(lam)])
            xk = np.array([x0 + sigma * y[i] for i in range(lam)])
            xk = np.clip(xk, -5.0, 5.0)
            # evaluate
            fk = np.empty(lam)
            for i in range(lam):
                fk[i] = func(xk[i])
                state["calls"] += 1
                if state["calls"] >= budget or time.time() > deadline:
                    break
            # if we ran out of budget mid-generation, fill the rest with inf
            if state["calls"] >= budget:
                for i in range(i + 1, lam):
                    fk[i] = np.inf
            # update best
            order = np.argsort(fk)
            if fk[order[0]] < best_f:
                best_f = fk[order[0]]
                best_x = xk[order[0]].copy()
                no_improve = 0
            else:
                no_improve += 1

            # weighted mean of mu best
            xnew = np.zeros(dim)
            for i in range(mu):
                xnew += w[i] * xk[order[i]]

            # update covariance square root (rotation)
            yinv = np.array([Cinv_sqrt @ (xk[order[i]] - x0) / sigma for i in range(mu)])
            v = np.zeros((dim, dim))
            for i in range(mu):
                v += np.outer(w[i] * yinv[i], yinv[i])
            Csqrt = (1 - c_1 - c_mu) * Csqrt + c_mu * (Csqrt @ v + v @ Csqrt) + c_1 * (
                (1 - c_c) * Csqrt + c_c * (xnew - x0) / sigma
            )
            # recompute Cinv_sqrt via Cholesky
            C = Csqrt @ Csqrt.T
            C += 1e-12 * np.eye(dim)  # numerical safety
            try:
                L = np.linalg.cholesky(C)
                Cinv_sqrt = np.linalg.inv(L).T
            except np.linalg.LinAlgError:
                C = np.linalg.pinv(C) + 1e-8 * np.eye(dim)
                Cinv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-6 * np.eye(dim))).T

            # path covariance
            pc = (1 - c_c) * pc + c_c * (xnew - x0) / sigma
            # path sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_c * (2 - c_c) * mueff) * Cinv_sqrt @ (xnew - x0) / sigma
            # step size adaptation
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e3)

            x0 = xnew

            # stagnation restart
            if no_improve > max(150, 20 * dim):
                break

        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        state = {"calls": 0}
        # wall-clock deadline with safety margin (evaluator kills at 330s)
        deadline = time.time() + 300.0

        best_x = None
        best_f = np.inf

        # multi-restart CMA-ES: first a broad start, then restarts from best
        n_restarts = 4
        budget_per = max(10, budget // n_restarts)
        for r in range(n_restarts):
            if state["calls"] >= budget or time.time() > deadline:
                break
            if r == 0:
                x0 = rng.uniform(-5, 5, dim)
                sigma = 2.0
            else:
                # restart near current best with a fresh, larger sigma
                x0 = best_x + rng.standard_normal(dim) * 0.5
                x0 = np.clip(x0, -5, 5)
                sigma = 1.5
            bx, bf = self._cmaes(func, x0, sigma, budget, deadline, rng, state)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        if best_x is None:
            x0 = np.zeros(dim)
            best_f = func(x0)
            state["calls"] += 1
            best_x = x0

        return best_x, best_f
# EVOLVE-BLOCK-END
