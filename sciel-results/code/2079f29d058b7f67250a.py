# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) on [-5,5]^dim.

    Structural change vs the (1+1)-ES seed: instead of isotropic Gaussian steps
    with a scalar sigma, this maintains a full covariance matrix C that is
    adapted to the local curvature. On the bent-cigar (f12) this lets the
    sampling ellipsoid align with the elongated valley, so progress along the
    narrow direction is not throttled by the wide direction. A population of
    lambda offspring is sampled each generation, the top-mu are weighted, and
    the mean, step size and covariance are updated with path-based adaptation
    (rank-1 and rank-mu). Time-guarded restarts hedge against a bad initial
    mean on the unknown shifted/rotated minimum.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 285.0   # safety margin under the 330s kill

        # ---- global best across all restarts (return interface unchanged) ----
        best_x = np.array([0.0] * dim)
        best_f = np.inf

        def evaluate(x):
            return float(func(np.clip(x, -5.0, 5.0)))

        def cma_run(x0, budget, seed):
            nonlocal best_x, best_f
            rng = np.random.default_rng(seed)

            # population size
            lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))
            if lam > 3 * dim:
                lam = 3 * dim
            mu = lam // 2

            # weights (variance-effectiveness-normalized)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            # adaptation parameters (standard CMA-ES choices)
            c_sigma = (mueff + 2.0) / (dim + mueff + 5.0)
            d_sigma = 1.0 + 2.0 * max(0.0, (mueff - 1.0) / (dim + 1.0)) + c_sigma
            c_c = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
            c_1 = (2.0 - mueff / (dim + 2.0) + 2.0 * mueff / (dim + 1.0)) / (
                (dim + 1.3) ** 2 + mueff
            )
            c_mu = min(
                1.0 - c_1,
                2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff),
            )
            chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

            x = np.array(x0, dtype=float)
            sigma = 2.0
            C = np.eye(dim)
            Cinv = np.eye(dim)
            sqrtC = np.eye(dim)
            sqrtCinv = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            f_best = np.inf
            x_best = x.copy()
            calls = 0

            def refresh_sqrt():
                # recompute matrix square roots via SVD (stable, dim is tiny)
                nonlocal sqrtC, sqrtCinv, Cinv
                U, s, Vt = np.linalg.svd(C)
                sqrtC = (U * np.sqrt(s)) @ Vt
                Cinv = (U * (1.0 / s)) @ Vt
                sqrtCinv = (U * (1.0 / np.sqrt(s))) @ Vt

            refresh_sqrt()

            while calls < budget:
                if time.time() > deadline:
                    break

                # sample population
                z = rng.standard_normal((lam, dim))
                pop = x[None, :] + sigma * (z @ sqrtC)
                pop = np.clip(pop, -5.0, 5.0)

                vals = np.empty(lam)
                for i in range(lam):
                    vals[i] = evaluate(pop[i])
                    calls += 1
                    if time.time() > deadline:
                        break
                else:
                    pass
                if time.time() > deadline:
                    # finish the population evaluation to avoid a torn update
                    for i in range(lam):
                        if vals[i] == vals[i]:  # already set
                            pass
                        else:
                            vals[i] = evaluate(pop[i])
                            calls += 1
                    if time.time() > deadline:
                        break

                order = np.argsort(vals)
                top = order[:mu]
                for i in top:
                    if vals[i] < f_best:
                        f_best = vals[i]
                        x_best = pop[i].copy()
                    if vals[i] < best_f:
                        best_f = vals[i]
                        best_x = pop[i].copy()

                y_w = np.sum(w[:, None] * pop[top], axis=0)
                d = y_w - x

                # update evolution paths
                Cinv_sqrtC_d = sqrtCinv @ (sqrtC @ d)
                ps = (1.0 - c_sigma) * ps + np.sqrt(c_sigma * (2.0 - c_sigma) * mueff) * Cinv_sqrtC_d
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * (calls + 1)))) < (
                    1.4 + 2.0 / (dim + 1.0)
                ) * chiN
                pc = (1.0 - c_c) * pc + hsig * np.sqrt(c_c * (2.0 - c_c) * mueff) * d

                # update covariance
                delta_hsig = (1.0 - hsig) * c_c * (2.0 - c_c)
                rank1 = np.outer(pc, pc)
                rankmu = np.sum(w[:, None, None] * np.outer(pop[top] - x, pop[top] - x), axis=0)
                C = (1.0 + c_sigma * (2.0 - c_sigma) * mueff * (hsig - 1.0) + c_1 + c_mu - c_1 - c_mu) * C
                C = C + c_1 * (rank1 - C)
                C = C + c_mu * (rankmu - C)
                C = 0.5 * (C + C.T)

                # adapt step size
                sigma = sigma * np.exp(
                    c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1.0)
                )
                sigma = min(max(sigma, 1e-12), 1e3)

                # move the mean
                x = x + d

                # keep C well-conditioned; refresh sqrt every few generations
                tr = np.trace(C) / dim
                if tr > 0:
                    C = C / tr * tr  # no-op, keep for clarity
                if (calls // lam) % 5 == 0:
                    refresh_sqrt()

            return x_best, f_best, calls

        # ---- multi-start: spread initial means over the box, then CMA-ES ----
        # A few diverse restarts hedge against the unknown shift; the global
        # best is tracked so later restarts only need to beat what we have.
        base_rng = np.random.default_rng(12345)
        n_starts = 4
        per = max(1, (self.budget - n_starts) // n_starts)
        used = 0
        for s in range(n_starts):
            if time.time() > deadline:
                break
            if s == 0:
                x0 = np.zeros(dim)
            else:
                x0 = base_rng.uniform(-4.5, 4.5, dim)
            # seed each restart with a distinct stream
            x_best, f_best, c = cma_run(x0, per, seed=1000 + 7 * s)
            used += c
            if time.time() > deadline:
                break

        # spend any remaining budget on a final polish around the global best
        remaining = self.budget - used
        if remaining > 4 and best_f < np.inf and time.time() < deadline:
            cma_run(best_x, remaining, seed=99999)

        return best_x, best_f
# EVOLVE-BLOCK-END
