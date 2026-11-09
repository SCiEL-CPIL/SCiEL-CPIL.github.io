"""Rewritten optimizer for BBOB f15 (Rastrigin rotated).

Structural change from the seed (1+1)-ES: a population-based CMA-ES with
recombination. Rastrigin is heavily multimodal, so a single-lineage (1+1)-ES
gets trapped in local minima; a population with recombination can escape them.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with recombination + a multimodal restart layer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0   # safety margin below the 300 s kill
        rng = np.random.default_rng(12345)
        d = self.dim
        lam = 8

        def evalf(x):
            return float(func(np.asarray(x, dtype=float)))

        def cmaes(x0, sigma0, remaining):
            n = d
            x = x0.astype(float)
            sigma = sigma0
            C = np.eye(n)
            Cs = np.zeros(n)
            ps = np.zeros(n)
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / w.sum()
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            pop = [x + sigma * B @ rng.standard_normal(n) for _ in range(lam)]
            vals = [evalf(p) for p in pop]
            order = np.argsort(vals)
            best_x = pop[order[0]].copy()
            best_f = vals[order[0]]

            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.array(pop[order[:mu]])
                yw = (w[:, None] * y).sum(axis=0)
                m = x + sigma * (B @ yw)

                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chiN * (B @ (yw / sigma))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * chiN:
                    Cs = (1 - c_c) * Cs + c_c * (2 - c_c) * (np.outer(ps, ps) - C)

                dC = c_1 * (np.outer(Cs, Cs) - C) + c_mu * sum(w[i] * np.outer(yw[i], yw[i]) for i in range(mu))
                C = C + dC
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))
                sigma = min(sigma, 10.0)

                x = m
                D, B = np.linalg.eigh(C)
                D = np.clip(np.sqrt(np.maximum(D, 1e-20)), 1e-10, 1e10)
                B = B * D

                pop = [x + sigma * B @ rng.standard_normal(n) for _ in range(lam)]
                vals = [evalf(p) for p in pop]
                order = np.argsort(vals)
                if vals[order[0]] < best_f:
                    best_f = vals[order[0]]
                    best_x = pop[order[0]].copy()
            return best_x, best_f

        def restarts(remaining):
            n = d
            # seed a diverse initial population (helps early multimodal progress)
            seeds = [np.clip(rng.uniform(-4.5, 4.5, n), -5, 5) for _ in range(lam)]
            sf = [evalf(s) for s in seeds]
            sidx = np.argsort(sf)
            best_x = seeds[sidx[0]].copy()
            best_f = sf[sidx[0]]

            # restart budget: spend ~70% of the remaining budget on the first
            # attempt, shrinking each restart; the last attempt gets the rest.
            b = int(remaining * 0.70)
            while remaining > 0 and b >= 1:
                x0 = best_x + rng.standard_normal(n) * 0.5
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.2
                xb, fb = cmaes(x0, sigma0, b)
                if fb < best_f:
                    best_f = fb
                    best_x = xb
                remaining -= b
                if time.time() > deadline:
                    break
                b = max(1, int(remaining * 0.70))
            return best_x, best_f

        return restarts(self.budget)
# EVOLVE-BLOCK-END
