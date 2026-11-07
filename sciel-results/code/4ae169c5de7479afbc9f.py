"""Optimizer for BBOB-style f15 (Rastrigin rotated) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard for 5D Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin

        rng = np.random.default_rng(123)
        d = self.dim

        # Initialize at origin
        x = np.zeros(d)
        fx = func(x)
        calls = 1

        # Quick multi-start to find a good region
        for _ in range(5):
            if calls >= self.budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4, 4, d)
            fx_try = func(x_try)
            calls += 1
            if fx_try < fx:
                x, fx = x_try, fx_try

        best_x = x.copy()
        best_f = fx

        # CMA-ES setup
        sigma = 1.0
        lam = max(4 + int(3 * np.log(d)), 8)
        c1 = (4 + np.log(d)) / (d**1.6 + 4 + 2 * np.log(d))
        cmu = min(1 - c1, 2 * (d - 2 + 1.0/d) / ((d + 2)**2 + 2))
        cs = (d + 1.3) / (d + 5)
        ds = 1 + 2 * max(0.0, np.sqrt((lam - 1) / (d + 1)) - 1) + cs
        chi = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))

        pc = np.zeros(d)
        ps = np.zeros(d)
        C = np.eye(d)

        while calls + lam <= self.budget:
            if time.time() > deadline:
                break

            # Generate lambda samples
            L = np.linalg.cholesky(C + 1e-10 * np.eye(d))
            Z = rng.standard_normal((lam, d))
            Y = x[None, :] + sigma * (Z @ L.T)
            Y = np.clip(Y, -5, 5)

            # Evaluate
            F = np.array([func(yi) for yi in Y])
            calls += lam

            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            mu = lam // 2

            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()

            # Weights
            w = np.zeros(mu)
            w[:mu] = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w**2)

            # Mean update
            ymean = np.sum(w[:, None] * Y[:mu], axis=0)

            # Step size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * ymean
            hsig = min(1.0, (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi + 1e-15) < (1.4 + 2 / (d + 1)))
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * ymean / sigma

            # Covariance update
            delta = 1 + (1 - hsig) * c1 * (2 - c1) + (cmu - c1) * (mueff - 1)
            rank1 = np.outer(pc, pc)
            rankmu = np.sum(w[:, None, None] * np.outer(Y[:mu] - x, Y[:mu] - x) / sigma**2, axis=0)
            C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu / (1 - np.trace(C) / d) * (d + 1)

            # Ensure C is symmetric and SPD
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update mean
            x = x + ymean
            x = np.clip(x, -5, 5)

            # Sigma update
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
