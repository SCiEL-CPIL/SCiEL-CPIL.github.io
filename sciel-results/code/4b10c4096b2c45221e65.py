# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with restarts for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = 4 * dim  # population size
        # Initial step size
        sigma = 1.0
        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition cache
        D_inv = np.ones(dim)  # 1/sqrt(eigenvalues)
        B = np.eye(dim)
        # Update counters
        ds = 1.0 / (1 + 0.3 + 2.0 / (lambda_ - 2))  # sigma adaptation rate
        dcs = (lambda_ + 2) / (lambda_ + 8)  # path length adaptation rate
        cc = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        cmu1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        cmu2 = min(1 - cmu1, 2 / ((dim + 2) ** 2 + lambda_))
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + lambda_)
        c1 = min(c1, 2 * (1 - cmu1 - cmu2))

        # Path variables
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Best solution tracking
        best_x = np.full(dim, 0.0)
        best_f = float('inf')
        calls_made = 0

        def time_check():
            return time.time() > deadline

        # Main loop
        while calls_made < budget and not time_check():
            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.randn(dim)
                # Transform using B and D_inv
                y = B @ (D_inv * z)
                pop[i] = y

            # Evaluate population
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                x = np.clip(best_x + sigma * pop[i], -5, 5)
                fvals[i] = func(x)
                calls_made += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x.copy()
                if time_check():
                    break

            if time_check():
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Weighted mean
            weights = np.log(lambda_ / 2 + 0.5) - np.log(np.arange(1, lambda_ + 1))
            weights /= weights.sum()
            mueff = int(np.sum(weights[:int(np.sqrt(lambda_))]) ** 2)

            # Compute new mean
            new_mean = np.zeros(dim)
            for i in range(lambda_):
                new_mean += weights[i] * pop[i]

            # Update path ps
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * mueff) / np.linalg.norm(B.T @ (D_inv * new_mean)) * (B.T @ (D_inv * new_mean))

            # Update path pc
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - ds) ** (2 * (calls_made // lambda_))) < (1.4 + 2 / (dim + 1)) * (1 - ds)
            pc = (1 - ccs) * pc + hsig * np.sqrt(ccs * (2 - ccs) * mueff) * (B.T @ (D_inv * new_mean))

            # Update covariance matrix
            delta = 0 if hsig else 1
            C = (1 + c1 * delta + c1 * (cmu1 + cmu2) - c1 - cmu1 - cmu2) * C
            for i in range(int(mueff)):
                v = pop[i] - new_mean
                C += cmu1 * np.outer(v, v)
            for i in range(int(mueff)):
                v = pop[i] - new_mean
                C += cmu2 * np.outer(v, v)
            C += c1 * (np.outer(pc, pc) + delta * C - np.eye(dim))

            # Regularization to maintain positive semi-definiteness
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                D_inv = 1.0 / np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                D_inv = np.ones(dim)
                B = np.eye(dim)

            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            sigma *= np.exp(ds / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update mean
            best_x = np.clip(best_x + new_mean * sigma, -5, 5)

            # Check for stagnation - restart if no improvement in last 200 calls
            if calls_made > 100:
                # Simple stagnation check: if sigma is very small or very large, restart
                if sigma < 1e-6 or sigma > 50:
                    # Restart with new random center
                    center = np.random.uniform(-4, 4, dim)
                    best_x = center
                    best_f = func(best_x)
                    calls_made += 1
                    sigma = 1.0
                    C = np.eye(dim)
                    D_inv = np.ones(dim)
                    B = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    if time_check():
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
