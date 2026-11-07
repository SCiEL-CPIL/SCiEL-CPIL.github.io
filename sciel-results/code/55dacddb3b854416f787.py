"""Optimized BBOB f14 solver with CMA-ES-like strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimization for BBOB f14 (Different powers) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Phase 1: Broad exploration with multiple random starts
        n_initial = min(20, budget)
        best_x = None
        best_f = np.inf

        for i in range(n_initial):
            if time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = rng.uniform(-2, 2, dim)
            else:
                x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        calls_made = n_initial

        if calls_made >= budget:
            return best_x, best_f

        # Phase 2: CMA-ES-like optimization
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        C = np.eye(dim)  # covariance matrix
        pc = np.zeros(dim)  # evolution path for covariance
        ps = np.zeros(dim)  # evolution path for step-size control

        # CMA-ES parameters
        lambda_ = 4 * int(np.floor(3 + np.log(dim)))  # population size
        mu = lambda_ // 2  # number of weights
        weights = np.log(np.arange(mu, 0, -1)) - np.log(mu)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        while calls_made < budget and time.time() < deadline:
            # Generate lambda_ candidates
            Y = rng.standard_normal((lambda_, dim))
            # Transform using Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(dim)
                L = np.eye(dim)
            
            Y = Y @ L.T
            X = x + sigma * Y
            X = np.clip(X, -5, 5)
            
            # Evaluate all candidates
            F = []
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                F.append(func(X[i]))
                calls_made += 1
                if calls_made >= budget:
                    break
            
            if len(F) < lambda_:
                break
            
            F = np.array(F)
            
            # Sort by fitness
            order = np.argsort(F)
            X_best = X[order[:mu]]
            F_best = F[order[:mu]]
            
            # Update best
            if F_best[0] < fx:
                fx = F_best[0]
                x = X_best[0].copy()
            
            # Update mean
            x_new = x + np.sum(weights[:, None] * (X_best - x), axis=0)
            
            # Update step size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * L.T @ (x_new - x) / sigma
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x) / sigma
            delta_sig = (1 - c_c) - c_c * (np.dot(pc, pc) / dim - 1)
            C = (1 + c_1 * delta_sig - c_1 - c_mu) * C + c_1 * np.outer(pc, pc)
            C += c_mu * np.sum(weights[:, None] * np.outer(X_best - x, X_best - x) / sigma**2, axis=0)
            
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            x = x_new
            sigma = max(sigma, 0.01)
            sigma = min(sigma, 10.0)
            
            # Check for convergence
            if np.all(np.abs(eigvals) < 1e-12):
                break

        return x, fx
# EVOLVE-BLOCK-END
