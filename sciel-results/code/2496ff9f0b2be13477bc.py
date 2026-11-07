"""Fresh design: CMA-ES core + Rastrigin-snap refinement for f15 (rotated Rastrigin)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES on the rotated Rastrigin, plus a 2*pi-periodic 'snap' refinement.

    The CMA-ES adapts its covariance to the hidden rotation/shift so it can
    follow the elongated basin structure; the snap step exploits the fact that
    Rastrigin's per-axis minima sit on the 2*pi lattice to land precisely in a
    basin, which lowers the best-so-far both early and late (raising AOCC).
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0          # safety margin well under the 300s kill
        two_pi = 2.0 * np.pi

        def within_budget():
            return (budget > 0 and time.time() < deadline)

        def snap(x):
            """Move each coordinate to its nearest 2*pi lattice point (Rastrigin
            per-axis minima), then accept only if it improves the value."""
            xs = np.round(x / two_pi) * two_pi
            fx = func(xs)
            return xs, fx

        rng = np.random.default_rng(12345)

        # ---- Phase 1: coarse CMA-ES (adapts to the rotated basin structure) ----
        x0 = rng.uniform(-4.0, 4.0, dim)
        x = x0.copy()
        fx = func(x)
        budget -= 1

        sigma = 1.2
        lam = 4 + int(2 * np.ceil(3 * np.log(dim)))
        # CMA-ES weights / effective population size
        w = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
        w = w / w.sum()
        mu = int(np.ceil(lam / 2))
        mu_eff = 1.0 / np.sum(w[:mu] ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        Csqrt = np.eye(dim)
        Cinv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        hs = 0.0

        def cma_step():
            nonlocal x, fx, sigma, C, Csqrt, Cinv, p_sigma, p_c, hs
            budget_nonlocal = None
            # generate offspring
            ys = []
            for _ in range(lam):
                z = rng.multivariate_normal(np.zeros(dim), C, check_valid='ignore')
                y = x + sigma * z
                ys.append(y)
            ys = np.array(ys)
            fs = np.array([func(y) for y in ys])
            budget_nonlocal = lam
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            # update mean
            m_new = np.sum(w[:mu, None] * ys[:mu], axis=0)
            # sigma adaptation
            y_sigma = (Csqrt @ p_sigma + w[0] * (m_new - x)) / sigma
            hs = (np.linalg.norm(y_sigma) / (1 - (1 - c_sigma) ** (budget_nonlocal + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
            p_sigma = (1 - c_sigma) * p_sigma + hs * c_sigma / c_sigma * y_sigma * c_sigma / c_sigma  # placeholder
            # proper p_sigma update:
            p_sigma = (1 - c_sigma) * p_sigma + (c_sigma if hs else 0.0) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(y_sigma) / chi_n - 1))
            sigma = min(max(sigma, 1e-12), 10.0)
            # C adaptation
            y_c = (Csqrt @ p_c + w[0] * (m_new - x)) / sigma
            p_c = (1 - c_c) * p_c + (c_c if hs else 0.0) * y_c
            delta_hs = (1 - hs) * 2 / (dim + 1) - 1
            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) - C) \
                + c_mu * np.sum(w[:mu, None, None] * np.outer(ys[:mu], ys[:mu]) / sigma ** 2, axis=0)
            # recompute sqrt / inv via eig
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-14, None)
            Csqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
            Cinv = evecs @ np.diag(1.0 / evals) @ evecs.T
            # accept best
            x, fx = m_new, fs[0]
            return budget_nonlocal

        # run coarse CMA-ES for a fixed number of generations (bounded)
        gens = 18
        for _ in range(gens):
            if not within_budget():
                break
            used = cma_step()
            budget -= used
            # periodic snap refinement: try to land on the 2*pi lattice
            xs, fsnap = snap(x)
            if fsnap < fx:
                x, fx = xs, fsnap
            # keep best-so-far; occasionally re-seed sigma if stuck
            if budget <= 0:
                break

        # ---- Phase 2: fine CMA-ES around best point ----
        if within_budget() and budget > 0:
            sigma = min(sigma, 0.4)
            lam2 = 4 + int(2 * np.ceil(3 * np.log(dim)))
            w = np.log(lam2 / 2 + 0.5) - np.log(np.arange(1, lam2 + 1))
            w = w / w.sum()
            mu = int(np.ceil(lam2 / 2))
            mu_eff = 1.0 / np.sum(w[:mu] ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            C = np.eye(dim)
            Csqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            hs = 0.0

            def cma_step2():
                nonlocal x, fx, sigma, C, Csqrt, p_sigma, p_c, hs
                ys = np.array([x + sigma * rng.multivariate_normal(np.zeros(dim), C, check_valid='ignore')
                               for _ in range(lam2)])
                fs = np.array([func(y) for y in ys])
                order = np.argsort(fs)
                ys = ys[order]
                fs = fs[order]
                m_new = np.sum(w[:mu, None] * ys[:mu], axis=0)
                y_sigma = (Csqrt @ p_sigma + w[0] * (m_new - x)) / sigma
                hs = (np.linalg.norm(y_sigma) / (1 - (1 - c_sigma) ** (lam2 + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
                p_sigma = (1 - c_sigma) * p_sigma + (c_sigma if hs else 0.0) * y_sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(y_sigma) / chi_n - 1))
                sigma = min(max(sigma, 1e-12), 5.0)
                y_c = (Csqrt @ p_c + w[0] * (m_new - x)) / sigma
                p_c = (1 - c_c) * p_c + (c_c if hs else 0.0) * y_c
                delta_hs = (1 - hs) * 2 / (dim + 1) - 1
                C = (1 + c_1 * delta_hs - c_1 - c_mu) * C \
                    + c_1 * (np.outer(p_c, p_c) - C) \
                    + c_mu * np.sum(w[:mu, None, None] * np.outer(ys[:mu], ys[:mu]) / sigma ** 2, axis=0)
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-14, None)
                Csqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
                x, fx = m_new, fs[0]
                return lam2

            for _ in range(40):
                if not within_budget():
                    break
                used = cma_step2()
                budget -= used
                xs, fsnap = snap(x)
                if fsnap < fx:
                    x, fx = xs, fsnap
                if budget <= 0:
                    break

        # ---- Phase 3: cheap local (1+1)-ES polish until budget/time runs out ----
        if within_budget() and budget > 0:
            sigma = 0.15
            for _ in range(budget):
                if not within_budget():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                budget -= 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9
                # occasional snap
                if budget % 25 == 0:
                    xs, fsnap = snap(x)
                    if fsnap < fx:
                        x, fx = xs, fsnap

        return x, fx
# EVOLVE-BLOCK-END
