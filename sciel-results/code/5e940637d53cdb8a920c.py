"""Black-box optimizer for BBOB f14 (Different powers): population CMA-ES."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


class Optimizer:
    """Population-based CMA-ES with step-size adaptation and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin under 300 s limit

        # --- initialize population around the box center ---
        x0 = np.zeros(dim)
        sigma = 4.0
        pop = np.random.default_rng(12345).normal(x0, sigma, size=(5, dim))
        pop = np.clip(pop, -5.0, 5.0)

        fx = np.array([func(p) for p in pop], dtype=float)
        best_i = int(np.argmin(fx))
        x_best = pop[best_i].copy()
        f_best = float(fx[best_i])
        used = len(fx)

        # CMA state
        mu = 2
        w = np.exp(-1.0 / (2.0 * mu))
        w = w / w.sum()
        w = np.maximum(w, 1e-12)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x_mean = np.average(pop, axis=0, weights=w)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        invsqrtC = np.eye(dim)

        def invsqrt(matrix):
            eigval, eigvec = np.linalg.eigh(matrix)
            eigval = np.clip(eigval, 1e-12, None)
            invsqrt_eig = 1.0 / np.sqrt(eigval)
            return eigvec @ (invsqrt_eig[:, None] * eigvec.T)

        invsqrtC = invsqrt(C)

        def _eigstep():
            nonlocal C, invsqrtC
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.clip(eigval, 1e-12, None)
            C = (eigvec * eigval) @ eigvec.T
            invsqrtC = eigvec @ (1.0 / np.sqrt(eigval)[:, None] * eigvec.T)

        calls_left = budget - used
        gen = 0
        while used < budget:
            if time.time() > deadline:
                break
            # sample population
            z = np.random.default_rng().standard_normal((5, dim))
            newpop = np.clip(x_mean + sigma * (z @ invsqrtC.T), -5.0, 5.0)
            fnew = np.array([func(p) for p in newpop], dtype=float)
            used += 5

            # track best
            li = int(np.argmin(fnew))
            if fnew[li] < f_best:
                f_best = float(fnew[li])
                x_best = newpop[li].copy()

            # sort by fitness
            order = np.argsort(fnew)
            y = newpop[order]
            yw = y[:mu]

            # new mean
            x_mean = np.average(yw, axis=0, weights=w)

            # update p_sigma
            dvec = (invsqrtC @ (x_mean - (x_mean - sigma * p_sigma / c_sigma))) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * dvec

            # h_sigma
            hs = (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) / chi_n
            hs = 1.0 if hs < 1.17 else 0.0

            # update p_c
            p_c = (1 - c_c) * p_c + hs * math.sqrt(c_c * (2 - c_c) * mu_eff) * (invsqrtC @ (x_mean - (x_mean - p_c / c_c))) / sigma

            # rank-mu update
            delta_h = (1.7 + 0.5 / dim) * (1 - hs)
            delta_h = delta_h if hs < 1.17 else 0.0
            yd = (yw - x_mean + p_c / c_c) / sigma
            C = ((1 - (c_1 + c_mu)) * C
                 + c_1 * (np.outer(p_c, p_c) + (delta_h - 1) * C)
                 + c_mu * np.sum(w[:, None, None] * yd[:, None, :] * yd[:, None, :].T, axis=0))

            # step-size adaptation (CSA)
            sigma *= math.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-6, 50.0))

            # periodic recompute of inverse sqrt
            if (gen + 1) % 10 == 0:
                _eigstep()

            gen += 1

        return x_best, f_best
# EVOLVE-BLOCK-END
