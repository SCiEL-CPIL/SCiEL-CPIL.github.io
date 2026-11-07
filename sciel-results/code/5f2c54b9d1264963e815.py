"""Optimizer for BBOB-style f11 (Discus) using CMA-ES with proper implementation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill

        d = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            return float(func(x))

        def alive():
            return calls[0] < budget and time.time() < deadline

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float).copy()

        rng = np.random.default_rng(42)

        # ---- Phase 0: diverse initial sampling ----
        candidates = [np.zeros(d)]
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, d))
        for axis in range(d):
            for val in [-3.0, 3.0]:
                pt = np.zeros(d)
                pt[axis] = val
                candidates.append(pt)

        for c in candidates:
            if not alive():
                break
            fc = wrapped(c)
            update_best(c, fc)

        # ---- Proper CMA-ES implementation ----
        def cma_es(x0, sigma0, max_calls):
            nonlocal best_f, best_x
            n = d
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = (2.0 / ((n + 1.8) ** 2 + mueff)) * (1.0 - cc)
            cmu = min(1.0 - c1, (2.0 * (mueff - 2.0 + 1.0 / mueff)) / ((n + 2.0) ** 2 + mueff))

            xmean = np.array(x0, dtype=float).copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            calls_used = 0
            no_improve = 0

            while alive() and calls_used < max_calls:
                # Generate population
                ys = rng.standard_normal((lam, n))
                xs = np.array([xmean + sigma * (ys[i] @ B @ D) for i in range(lam)])
                xs = np.clip(xs, -5, 5)

                # Evaluate
                fs = np.array([wrapped(xs[i]) for i in range(lam)])
                calls_used += lam

                # Sort
                idx = np.argsort(fs)
                xs = xs[idx]
                fs = fs[idx]
                ys = ys[idx]

                # Update best
                old_best = best_f
                update_best(xs[0], fs[0])
                if best_f < old_best:
                    no_improve = 0
                else:
                    no_improve += 1

                # Convergence checks
                if fs[0] - fs[-1] < 1e-14 * max(1.0, abs(fs[0])):
                    break
                if sigma < 1e-12:
                    break
                if no_improve > 15:
                    break

                # Weighted recombination
                y_rec = np.sum(weights[:, None] * ys[:mu], axis=0)

                # Update step size control vector
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) / (D @ B).T @ y_rec
                hsig = (1.4 + 1.0 / n) * chi_n
                if np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls_used + 1) / (n + 1))) < hsig:
                    pc = (1.0 - cc) * pc + cc * y_rec
                    rank_pc = np.outer(pc, pc)
                else:
                    rank_pc = (1.0 + 2.0 * (cs / (1.0 - cs) - 1.0)) * np.outer(pc, pc)

                # Rank-mu update
                rank_mu = np.sum(weights[:, None] * (ys[:mu] * ys[:mu].T), axis=0)

                # Update covariance matrix
                C = (1.0 - c1 * cc * (1.0 - cc) - c1 - cmu) * C + c1 * rank_pc + cmu * rank_mu

                # Update mean
                xmean = xmean + sigma * (B @ D) @ y_rec
                xmean = np.clip(xmean, -5, 5)

                # Update step size (CSA)
                sigma *= np.exp(cs / (n + 1) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(sigma, 1e4)
                sigma = max(sigma, 1e-10)

                # Update B and D from C
                D2, B = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-16))
                D = D2

            return xmean, best_f

        # Run CMA-ES from best point
        if alive():
            x0 = best_x.copy() if best_x is not None else np.zeros(d)
            cma_es(x0, 1.0, budget - calls[0] - 20)

        # If still have budget, try additional restarts
        restarts = 0
        while alive() and calls[0] < budget - 15 and restarts < 5:
            restarts += 1
            # Perturb around current best
            sigma_restart = max(0.1, 1.0 / restarts)
            x0 = best_x + rng.normal(0, sigma_restart, d)
            x0 = np.clip(x0, -5, 5)
            f0 = wrapped(x0)
            update_best(x0, f0)

            # Run short CMA-ES
            if alive():
                cma_es(x0, sigma_restart, min(80, budget - calls[0] - 10))

        # Final local refinement using coordinate descent
        if alive() and calls[0] < budget - 5:
            x = best_x.copy()
            f = best_f
            step = 0.5
            for _ in range(20):
                if not alive():
                    break
                improved = False
                for i in range(d):
                    for direction in [-1, 1]:
                        x_new = x.copy()
                        x_new[i] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        f_new = wrapped(x_new)
                        if f_new < f:
                            x = x_new
                            f = f_new
                            improved = True
                            update_best(x, f)
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
