"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for the rotated Rosenbrock valley + (1+1)-ES polish, with a
    wall-clock safety guard. Minimizes func on [-5, 5]^dim within budget calls."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0          # safety margin before the 330 s kill
        d = self.dim
        budget = self.budget

        def _elapsed_ok():
            return time.time() < deadline

        # ---------- CMA-ES ----------
        lam = max(10, 4 + int(3 * np.log(d)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2.0 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))

        mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi = np.sqrt(d) * (1 + 1 / (4 * d) - 1 / (32 * d ** 2))

        best_x = mean.copy()
        best_f = float("inf")
        evals = 0
        gen = 0

        def _sample_pop():
            nonlocal mean, sigma, C
            z = np.random.standard_normal((lam, d))
            try:
                chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-12 * np.eye(d)
                chol = np.linalg.cholesky(C)
            pop = mean[None, :] + sigma * (z @ chol.T)
            return np.clip(pop, -5, 5)

        while evals < budget and _elapsed_ok():
            pop = _sample_pop()
            fvals = np.empty(lam)
            for i in range(lam):
                if evals >= budget or not _elapsed_ok():
                    break
                fvals[i] = func(pop[i])
                evals += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()
            if evals >= budget or not _elapsed_ok():
                break
            n = evals // lam + 1
            order = np.argsort(fvals[:n * lam])
            pop = pop[order]
            fvals = fvals[order]

            mean_new = np.sum(weights[:, None] * pop[:mu], axis=0)
            ydiff = (mean_new - mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * ydiff
            hsig = (np.linalg.norm(ps) /
                    np.sqrt(1 - (1 - cs) ** (2 * (evals // lam + 1)))) \
                < (1.4 + 2 / (d + 1)) * chi
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ydiff
            y = (pop[:mu] - mean) / sigma
            rank_mu = np.sum(weights[:, None, None] *
                             y[:, :, None] * y[:, None, :], axis=0)
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * np.eye(d))
                 + cmu * rank_mu)
            C = (C + C.T) / 2
            sigma *= np.exp(cs / d * (np.linalg.norm(ps) / chi - 1))
            sigma = min(sigma, 10.0)
            mean = mean_new
            gen += 1
            # Plateau detection: hand off to polish if progress is tiny
            if gen > 5 and sigma < 0.05:
                break

        # ---------- (1+1)-ES polish from the current program ----------
        x = best_x.copy()
        fx = best_f
        sigma2 = max(sigma, 0.02)
        while evals < budget and _elapsed_ok():
            y = np.clip(x + sigma2 * np.random.standard_normal(d), -5, 5)
            fy = func(y)
            evals += 1
            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma2 *= 1.5
            else:
                sigma2 *= 0.87

        return best_x, best_f
# EVOLVE-BLOCK-END
