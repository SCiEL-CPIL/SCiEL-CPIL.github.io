"""Optimizer for BBOB-style f11 (Discus) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        d = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            return float(func(x))

        def alive():
            return calls[0] < budget and time.time() < deadline

        # CMA-ES Parameters
        lam = 8  # Population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        
        # Initial state
        x_mean = np.zeros(d)
        sigma = 2.0
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        C = np.eye(d)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        # Evaluate initial mean
        f_mean = wrapped(x_mean)
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()

        iter_count = 0
        while alive():
            # Compute B and D from C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = 1.0 / np.sqrt(eigvals)
            B = eigvecs
            
            # Sample
            Z = rng.standard_normal((lam, d))
            X = x_mean + sigma * (B @ (D[:, None] * Z).T).T
            
            # Evaluate
            F = np.array([wrapped(x) for x in X])
            
            # Update best
            for i in range(lam):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
            
            # Sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * X[:mu], axis=0)
            
            # Update sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ (D * (x_mean - x_mean_old) / sigma))
            sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = np.clip(sigma, 1e-12, 100)
            
            # Update C
            hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (iter_count + 1))) < (1 + 2 / (d + 1)) * chiN
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_mean - x_mean_old) / sigma
            
            y = (x_mean - x_mean_old) / sigma
            C = (1 + c1 * (2 - c1) + cmu) * C + c1 * np.outer(y, y)
            
            # Rank-mu update
            Y_mu = (X[:mu] - x_mean_old) / sigma
            C += cmu * np.sum(weights[:, None, None] * Y_mu[:, :, None] * Y_mu[:, None, :], axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            iter_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
