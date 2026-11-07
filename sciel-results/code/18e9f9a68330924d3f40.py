"""Structural change: population-based CMA-ES with multi-start + deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Population-based CMA-ES (with multi-start) for box-constrained minimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cmaes(self, func, x0, sigma0, n_eval, rng, deadline):
        dim = self.dim
        n = dim
        # population size
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # step size adaptation
        c_sigma = (mueff + 1) / (n + 2) / 5
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
        # position adaptation
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - mueff / (n + 1.8) + 2 * w[0]) / ((n + 2) ** 2 + mueff)
        c_mu = (2 * (mueff / (n + 2) - c_1)) / (
            (n + 1.3) ** 2 + mueff
        )
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        x = np.asarray(x0, dtype=float).copy()
        sigma = sigma0
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        # precompute C^0.5
        C_half = np.eye(n)
        C_invhalf = np.eye(n)
        minfeval = 0
        best_x = x.copy()
        best_f = func(x)
        minfeval += 1

        while minfeval < n_eval:
            # deadline guard
            if time.time() > deadline:
                break
            # sample population
            ys = np.empty((lam, n))
            for i in range(lam):
                y = x + sigma * C_half @ rng.standard_normal(n)
                ys[i] = np.clip(y, -5.0, 5.0)
            # evaluate
            fs = np.empty(lam)
            for i in range(lam):
                fs[i] = func(ys[i])
                minfeval += 1
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = ys[i].copy()
            # sort
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            # re-evaluate mean of selected (weights on selected)
            y_mean = np.sum(w[:, None] * ys[:mu], axis=0)
            # step size update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2 - c_sigma) / mueff
            ) * C_invhalf @ (C_half @ y_mean) / sigma
            if np.linalg.norm(p_sigma) / np.sqrt(
                1 - (1 - c_sigma) ** (2 * (minfeval - minfeval))
            ) < (1.4 + 2 / (n + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (
                    np.linalg.norm(p_sigma) / chi_n - 1
                ))
            # position update
            p_c = (1 - c_c) * p_c + c_c * (y_mean - x)
            # rank-one + rank-mu covariance update
            y_p = C_half @ p_c
            C = (1 + c_c * (2 - c_c) * c_1 - c_1) * C + c_1 * np.outer(y_p, y_p)
            Cm = np.sum(
                [w[i] * np.outer(C_half @ (ys[i] - y_mean), C_half @ (ys[i] - y_mean))
                 for i in range(mu)],
                axis=0,
            )
            C = (1 + c_mu * (2 - c_mu) * c_1 - c_1) * C + c_mu * Cm
            # recompute square root
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.clip(eigval, 1e-12, None)
            C_half = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T
            C_invhalf = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
            # move mean
            x = y_mean
            # termination heuristic: if sigma tiny
            if sigma < 1e-8:
                break
        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        rng = np.random.default_rng(12345)
        deadline = time.time() + 300.0  # safety margin under 330s kill
        budget = self.budget

        # multi-start: allocate budget across a few restarts
        # first start gets the majority; later restarts refine
        global_best_x = None
        global_best_f = np.inf
        starts = 3
        # budget split: decreasing
        splits = [int(budget * 0.6), int(budget * 0.25), int(budget * 0.15)]
        for s in range(starts):
            if time.time() > deadline:
                break
            if s == 0:
                x0 = rng.uniform(-4.5, 4.5, dim)
                sigma0 = 2.5
            else:
                # restart near current best with smaller sigma
                x0 = global_best_x + rng.standard_normal(dim) * 0.5
                x0 = np.clip(x0, -5.0, 5.0)
                sigma0 = max(0.3, 2.5 * (0.5 ** s))
            n_eval = splits[s]
            bx, bf = self._cmaes(func, x0, sigma0, n_eval, rng, deadline)
            if bf < global_best_f:
                global_best_f = bf
                global_best_x = bx.copy()
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
