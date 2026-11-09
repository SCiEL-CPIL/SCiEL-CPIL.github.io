"""Rewritten optimizer for BBOB f12 (Bent Cigar) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Bent Cigar on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290

        n = self.dim
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(n))
        pop_size = max(pop_size, 4)
        mu = pop_size // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 - cc + min(w[0], 2 / (n + 1))) / ((n + 1.3) ** 2 + cc)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + cc))

        # Initialize
        mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Initial evaluation
        pop = [np.clip(mean + sigma * rng.standard_normal(n), -5, 5) for _ in range(pop_size)]
        fits = np.array([func(p) for p in pop])
        call_count = pop_size

        best_x = pop[np.argmin(fits)].copy()
        best_f = fits.min()

        while call_count < self.budget:
            if time.time() > deadline:
                break

            # Sample
            pop = []
            for _ in range(pop_size):
                z = rng.standard_normal(n)
                try:
                    chol = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    chol = np.linalg.cholesky(C)
                y = mean + sigma * (chol @ z)
                pop.append(np.clip(y, -5, 5))

            fits = np.array([func(p) for p in pop])
            call_count += pop_size

            idx_best = np.argmin(fits)
            if fits[idx_best] < best_f:
                best_f = fits[idx_best]
                best_x = pop[idx_best].copy()

            order = np.argsort(fits)
            pop_sorted = [pop[i] for i in order]

            # Update mean
            mean = sum(w[i] * pop_sorted[i] for i in range(mu))

            # Weighted difference
            y_inv = sum(w[i] * (pop_sorted[i] - mean) for i in range(mu)) / sigma

            # Update ps
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ y_inv)

            # Update pc
            norm_ps = np.linalg.norm(ps)
            denom = np.sqrt(1 - (1 - cs) ** (2 * call_count / (n * cs)))
            if denom < 1e-10:
                denom = 1e-10
            hs = norm_ps / denom
            threshold = (1.4 + 2 / (n + 1)) * np.sqrt((1 - (1 - cs) ** 2) / (1 - (1 - cs) ** (2 * call_count / (n * cs))))
            if hs < threshold:
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * y_inv
            else:
                pc = (1 - cc) * pc

            # Update C
            delta_h = (1 - hs / (1.4 + 2 / (n + 1))) * cs * (2 - cs)
            C = (1 - c1 - cmu + delta_h) * C
            for i in range(mu):
                zi = (pop_sorted[i] - mean) / sigma
                C += cmu * w[i] * np.outer(zi, zi)

            C = (C + C.T) / 2
            C += 1e-10 * np.eye(n)

            # Update sigma
            sigma *= np.exp(0.33 * (mu / pop_size - 0.5))
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
