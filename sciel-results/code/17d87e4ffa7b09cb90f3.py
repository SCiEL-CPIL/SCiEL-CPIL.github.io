import numpy as np
import time

class Optimizer:
    """
    CMA-ES optimizer for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    CMA-ES is particularly well-suited for sharp ridge functions as it
    adapts the covariance matrix to the local landscape geometry.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global random search to find a good starting region
        n_random = 30
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: CMA-ES
        # Initialize CMA-ES parameters
        # Population size
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))
        # Initial step size
        sigma = 1.0
        # Mean vector
        mean = best_x.copy()
        # Covariance matrix (identity)
        C = np.eye(dim)
        # Eigen decomposition for efficient sampling
        D = np.ones(dim)
        B = np.eye(dim)
        # Step size control parameters
        cs = (lambda_pop + 2) / (dim + lambda_pop + 5)
        ds = 1 + 2 * max(0, (lambda_pop - 2) / (dim + 1) - 1) + cs
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) * (lambda_pop + 1.8))
        cmu = min(1 - c1, (2 + (lambda_pop / 2 - 2) / (dim + 2)) / ((dim + 2) * (lambda_pop / 2 + 1.8)))
        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Weighted recombination weights
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        # CMA-ES main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lambda_pop):
                if calls >= budget or time.time() > deadline:
                    break
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Transform to current distribution
                x = mean + sigma * (B @ (D * z))
                x = np.clip(x, -5.0, 5.0)
                f_val = safe_call(x)
                if f_val is None:
                    break
                offspring.append((x, f_val))

            if len(offspring) < 2:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[1])

            # Update best
            if offspring[0][1] < best_f:
                best_f = offspring[0][1]
                best_x = offspring[0][0].copy()

            # Take the best mu offspring
            selected = [offspring[i][0] for i in range(min(mu, len(offspring)))]
            if len(selected) < 2:
                break

            # Weighted recombination
            new_mean = np.zeros(dim)
            for i, x in enumerate(selected):
                new_mean += weights[i] * x
            new_mean /= np.sum(weights[:len(selected)])

            # Update covariance matrix
            v = new_mean - mean
            if np.linalg.norm(v) > 1e-16:
                # Update path p_sigma
                p_sigma = (1 - cs) * p_sigma + cs * np.sqrt(mueff / (dim * (1 - (1 - cs)**2))) * (B @ (D**2 @ (B.T @ v))) / sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2)) / chiN < (1.4 + 2 / (dim + 1))
                # Update path p_c
                p_c = (1 - cs) * p_c + h_sigma * cs * np.sqrt(mueff / (1 - (1 - cs)**2)) * v / sigma
                # Update covariance
                C = (1 + c1 * (2 - c1) * h_sigma - c1 - cmu) * C
                C += c1 * np.outer(v, v)
                for i, x in enumerate(selected):
                    diff = x - mean
                    C += cmu * weights[i] * np.outer(diff, diff)
                # Update step size
                sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)
                # Update mean
                mean = new_mean
                # Recompute eigen decomposition periodically
                # For efficiency, we'll do it every few iterations
                if calls % (lambda_pop * 5) < lambda_pop:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    D = np.sqrt(eigvals)
                    B = eigvecs
            else:
                mean = new_mean

            # Check for convergence
            if np.max(D) - np.min(D) < 1e-8 or sigma < 1e-10:
                break

        return best_x, best_f
