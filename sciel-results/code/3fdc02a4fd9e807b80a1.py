import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Cross-Entropy Method (CEM)
        n_cem = 350  # Number of evaluations for CEM phase
        n_samples = 50  # Samples per CEM iteration
        elite_frac = 0.2  # Keep top 20%
        n_elite = max(5, int(n_samples * elite_frac))

        # Initial distribution: mean at 0, large sigma to cover box
        mu = np.zeros(dim)
        sigma = 4.0  # Standard deviation
        # Use diagonal covariance initially, will adapt
        cov = sigma ** 2 * np.eye(dim)

        best_x = None
        best_f = np.inf
        all_evals = 0

        # CEM phase
        while all_evals < n_cem:
            if time.time() > deadline:
                break

            # Sample points from current distribution
            samples = []
            values = []
            for _ in range(n_samples):
                if all_evals >= n_cem:
                    break
                # Sample from multivariate normal
                x = mu + np.sqrt(np.diag(cov)) * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                samples.append(x)
                values.append(fx)
                all_evals += 1

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            if len(samples) == 0:
                break

            values = np.array(values)
            samples = np.array(samples)

            # Find elite set (top n_elite by lowest values)
            elite_indices = np.argsort(values)[:n_elite]
            elite_samples = samples[elite_indices]

            # Update mean and covariance
            new_mu = np.mean(elite_samples, axis=0)
            if n_elite > 1:
                new_cov = np.cov(elite_samples, rowvar=False)
                # Regularize to avoid singular matrix
                new_cov += 1e-6 * np.eye(dim)
            else:
                new_cov = cov * 0.5

            # Smooth update to avoid instability
            mu = 0.7 * new_mu + 0.3 * mu
            cov = 0.7 * new_cov + 0.3 * cov

            # Ensure covariance is positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 1e-8, None)
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Phase 2: Local refinement from best point
        if best_x is None:
            # Fallback: random search
            for _ in range(budget - all_evals):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                all_evals += 1
            return best_x, best_f

        # Local search: (1+1)-ES with adaptive step
        x = best_x.copy()
        fx = best_f
        sigma_local = 0.5
        remaining = budget - all_evals

        for _ in range(remaining):
            if time.time() > deadline:
                break

            y = x + sigma_local * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            all_evals += 1

            if fy < fx:
                x = y
                fx = fy
                sigma_local *= 1.3
            else:
                sigma_local *= 0.85

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
