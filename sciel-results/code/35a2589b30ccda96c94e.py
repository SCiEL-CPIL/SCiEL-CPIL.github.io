import numpy as np
import time


class Optimizer:
    """CMA-ES with population recombination (cross) and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        sigma = 1.0
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1)**1.7 + 2)
        cmu = min(1 - c1, (2 + (2.0 / (dim + 1.3) - 2)) / ((dim + 1)**1.7 + 5))
        cs = (dim + 1.7) / (dim + 9.5)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))

        # Initialize
        x0 = np.zeros(dim)  # Start at center; rotation/shift unknown but center is safe
        fx = func(x0)
        best_x = x0.copy()
        best_f = fx
        calls = 1

        C = np.eye(dim)
        C_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        damps = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs

        # Precompute weights for recombination (cross operator)
        weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        # Eigenvalue decomposition cache for C
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)

        for it in range(budget - 1):
            if time.time() > deadline:
                break

            # Update eigen decomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            D = np.diag(np.sqrt(np.maximum(eigvals, 1e-12)))
            C_inv = (eigvecs @ np.diag(1.0 / np.maximum(eigvals, 1e-12)) @ eigvecs.T)

            # Generate population
            Y = np.zeros((pop_size, dim))
            X = np.zeros((pop_size, dim))
            for i in range(pop_size):
                # B = eigvecs * D
                # z ~ N(0, I)
                z = np.random.randn(dim)
                # x = x0 + sigma * B * z
                step = sigma * (eigvecs @ (D @ z))
                X[i] = np.clip(x0 + step, -5, 5)
                Y[i] = step

            # Evaluate
            F = np.zeros(pop_size)
            for i in range(pop_size):
                if time.time() > deadline:
                    # Fill rest with best known to avoid errors, but we break after loop
                    F[i] = best_f
                    calls += 1
                    continue
                F[i] = func(X[i])
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()

            if time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Recombination (Cross): Weighted mean of top mueff individuals
            xnew = x0 + sigma * np.dot(weights[:int(mueff)], Y[:int(mueff)])
            # Note: Standard CMA-ES uses weighted mean of all, but often top half is sufficient for stability
            # Let's use standard weighted mean of all sorted Y for better performance
            xnew = x0 + sigma * np.dot(weights, Y)

            # Path updates
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (eigvecs.T @ (eigvecs @ (xnew - x0)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (it + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * (xnew - x0)

            # Covariance update
            delta_hsig = 1.0 - hsig
            C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) + delta_hsig * cs * (2 - cs) * C) \
                + cmu * np.sum(weights[i] * np.outer(Y[i], Y[i]) for i in range(pop_size))

            # Ensure positive semi-definiteness and symmetry
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Sigma update
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 100)

            # Move center
            x0 = xnew
            fx = func(x0)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

            if calls >= budget:
                break

        return best_x, best_f
