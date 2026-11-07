"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with multi-start strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng):
        dim = self.dim
        n = dim
        
        # CMA-ES parameters
        lambda_ = max(4, int(np.ceil(4 + 3 * np.log(n))))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = int(1 / weights.sum())
        
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1) - 1)) + cs
        hsig = (1 + 2 / (n + 1)) / damps
        chis = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        mean = np.array(x0, dtype=float)
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        count = 0

        # Initial evaluation
        fmean = func(np.clip(mean, -5, 5))
        count += 1

        best_x = np.clip(mean, -5, 5).copy()
        best_f = fmean

        while count < budget:
            # Generate lambda_ candidates
            candidates = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                candidates.append((y, z))

            # Evaluate
            fvals = np.array([func(y) for y, _ in candidates])
            count += lambda_

            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = candidates[min_idx][0].copy()

            # Sort
            order = np.argsort(fvals)
            y_sorted = [candidates[i][0] for i in order]
            z_sorted = [candidates[i][1] for i in order]

            # Update mean
            mean_old = mean.copy()
            mean = mean + sum(weights[i] * (y_sorted[i] - mean) for i in range(mu))

            # Update sigma
            pc = (1 - cc) * pc + cc * np.sqrt(cc / (2 - cc)) * (mean - mean_old) / (sigma * np.sqrt(1 - (1 - cc) ** 2))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chis * (B.T @ (mean - mean_old)) / sigma
            
            if np.linalg.norm(ps) / chis < (1.4 + 2 / (n + 1)) * hsig:
                C = (1 - c1 - cmu) * C + c1 * (2 - 2 * mueff / lambda_) * np.outer(pc, pc)
                for i in range(mu):
                    C += cmu * weights[i] * np.outer(z_sorted[i], z_sorted[i])
                
                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
                
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chis - 1))
                sigma = np.clip(sigma, 1e-10, 5.0)

        return best_x, best_f

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start CMA-ES with different initializations
        n_starts = 3
        budget_per_start = budget // n_starts
        remaining = budget - (n_starts - 1) * budget_per_start

        best_x = None
        best_f = np.inf

        # Different starting points and step sizes
        starts = [
            (np.zeros(dim), 2.0),
            (rng.uniform(-3, 3, dim), 1.5),
            (rng.uniform(-4, 4, dim), 1.0),
        ]

        for start in range(n_starts):
            if start < n_starts - 1:
                b = budget_per_start
            else:
                b = remaining

            x0, sigma0 = starts[start]
            x_res, f_res = self._cma_es(func, x0, sigma0, b, rng)

            if f_res < best_f:
                best_f = f_res
                best_x = x_res

        return best_x, best_f
# EVOLVE-BLOCK-END
