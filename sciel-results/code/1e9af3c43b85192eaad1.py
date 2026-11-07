"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start and boundary handling for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280

        call_count = [0]

        def counting_func(x):
            call_count[0] += 1
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return float(val)

        def cma_es(start_x, sigma_init, max_evals, rng, seed_offset=0):
            n = len(start_x)
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = (2 - mueff / (n + 1) ** 2 + 2 * mueff / n) / ((n + 2) ** 2 + mueff)
            c_mu = (2 * (mueff / (n + 1) ** 2 - 1) + c_1) / ((n + 2) ** 2 + mueff / (n + 1))

            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            x = np.array(start_x, dtype=np.float64)
            x_best = x.copy()
            f_best = counting_func(x)
            sigma = float(sigma_init)
            evaluations = 1

            C = np.eye(n)
            C_eigvals = np.ones(n)
            C_eigvecs = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)

            gen = 0
            while evaluations + lam <= max_evals and call_count[0] < budget:
                if time.time() > deadline:
                    break
                gen += 1

                # Generate and evaluate population
                ys = np.array([rng.standard_normal(n) for _ in range(lam)])
                # Transform using current covariance
                zs = (C_eigvecs.T @ (C_eigvals * ys.T)).T
                xs = x + sigma * zs
                fs = np.array([counting_func(xi) for xi in xs])
                evaluations += lam

                # Sort
                order = np.argsort(fs)
                xs = xs[order]
                fs = fs[order]
                zs = zs[order]

                if fs[0] < f_best:
                    f_best = fs[0]
                    x_best = xs[0].copy()

                # Weighted mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += w[i] * zs[i]

                # Update sigma path
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_w / np.sqrt(np.sum(C_eigvals ** 2) + 1e-30)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen))) / chi_n < 1.4 + 2 / (n + 1)

                # Update C path
                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (x_best - x) / sigma
                else:
                    pc = (1 - c_c) * pc

                # Update mean
                x = x + sigma * y_w

                # Update covariance
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 - c_1) * C + (c_1 + c_mu) * (
                    np.outer(ps, ps) + delta_hsig * np.outer(pc, pc)
                )
                for i in range(mu):
                    C += c_mu * w[i] * np.outer(zs[i], zs[i])
                C = (C + C.T) / 2

                # Periodic eigen decomposition
                if gen % 10 == 0:
                    try:
                        C_eigvals, C_eigvecs = np.linalg.eigh(C)
                        C_eigvals = np.clip(C_eigvals, 1e-14, 1e14)
                    except:
                        pass

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Boundary: pull back if out
                x = np.clip(x, -5.0, 5.0)

                if sigma < 1e-12 or sigma > 1e8:
                    break

            return x_best, f_best, evaluations

        rng = np.random.default_rng(12345)
        best_x = np.zeros(dim)
        best_f = np.inf

        # Start points: origin, a few random
        starts = [np.zeros(dim)]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))

        remaining = budget
        for si, sx in enumerate(starts):
            if call_count[0] >= budget or time.time() > deadline:
                break
            if si < 2:
                alloc = min(remaining, budget // 3)
            else:
                alloc = min(remaining, budget // 5)
            if alloc < 10:
                break

            sigma0 = 2.0 if si == 0 else 1.5
            x_r, f_r, used = cma_es(sx, sigma0, alloc, rng)
            remaining -= used

            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()

        # Final local refinement
        if best_f < 100 and remaining > 20 and call_count[0] < budget:
            x_r, f_r, used = cma_es(best_x, 0.1, min(remaining, 80), rng)
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
