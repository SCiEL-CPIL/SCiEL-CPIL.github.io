"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast population search: brief exploration, then CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, budget, dim, rng, start_mean, start_sigma, deadline):
        """Run CMA-ES on a fixed budget and return (best_x, best_f, evals_used)."""
        # Population size
        lam = 9 + 3 * int(np.floor(np.log10(4 * dim)))
        if lam < 10:
            lam = 10
        mu = lam // 2

        # Weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        mean = np.array(start_mean, dtype=float)
        sigma = float(start_sigma)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C_sqrt = np.eye(dim)

        best_x = mean.copy()
        best_f = np.inf
        evals = 0

        def over():
            return time.time() > deadline or evals >= budget

        while not over():
            # Sample population
            pop = np.zeros((lam, dim))
            for i in range(lam):
                if over():
                    break
                y = rng.standard_normal(dim)
                z = C_sqrt @ y
                pop[i] = np.clip(mean + sigma * z, -5, 5)
                evals += 1
            if over():
                break

            # Evaluate
            fit = np.full(lam, np.inf)
            for i in range(lam):
                if over():
                    break
                try:
                    fv = func(pop[i])
                except Exception:
                    fv = np.inf
                fit[i] = fv
                if fv < best_f:
                    best_f = fv
                    best_x = pop[i].copy()
            if over():
                break

            # Sort
            idx = np.argsort(fit)
            pop = pop[idx]

            # New mean
            mean_new = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Step vector in C^-1/2 space
            delta = (mean_new - mean) / sigma
            try:
                v = np.linalg.solve(C_sqrt, delta)
            except np.linalg.LinAlgError:
                v = delta

            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * v
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

            # Covariance update
            H = np.zeros((dim, dim))
            for i in range(mu):
                yi = (pop[i] - mean) / sigma
                H += weights[i] * np.outer(yi, yi)
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * np.eye(dim))
            C += c_mu * H
            C = (C + C.T) / 2

            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-12 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    break

            mean = mean_new
            if sigma < 1e-12:
                break

        return best_x, best_f, evals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin under 300s

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        evals = 0

        def over():
            return time.time() > deadline or evals >= budget

        # ---- Phase 1: quick exploration (fixed grid + a few random starts) ----
        # A coarse grid over [-5,5]^5 at ~3 points per axis to find a basin.
        grid = np.array([-4.0, 0.0, 4.0])
        pts = []
        for a in grid:
            for b in grid:
                for c in grid:
                    for d in grid:
                        for e in grid:
                            pts.append([a, b, c, d, e])
        # Add a few random points to break symmetry
        pts.extend(rng.uniform(-5, 5, (20, dim)).tolist())

        for p in pts:
            if over():
                break
            x = np.clip(np.asarray(p, dtype=float), -5, 5)
            fx = func(x)
            evals += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---- Phase 2: CMA-ES refinement from the best point ----
        if best_x is None or over():
            return best_x, best_f

        remaining = budget - evals
        if remaining > 0:
            start_sigma = 1.5
            bx, bf, used = self._cma_es(
                func, remaining, dim, rng, best_x, start_sigma, deadline
            )
            evals += used
            if bf < best_f:
                best_f = bf
                best_x = bx

        return best_x, best_f
# EVOLVE-BLOCK-END
