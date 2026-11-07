import numpy as np
import time

class Optimizer:
    """Hybrid DE + CMA-ES for Weierstrass f16 with rotation handling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: DE-based global exploration (first 60% of budget)
        n_de = int(budget * 0.6)
        pop_size = 20
        F = 0.8  # DE mutation factor
        CR = 0.9  # Crossover rate

        # Initialize population with Latin Hypercube sampling
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            X[:, d] = rng.permutation(pop_size) * (10.0 / pop_size) - 5.0 + rng.uniform(0, 10.0/pop_size, pop_size)
            X[:, d] = np.clip(X[:, d], -5, 5)

        # Evaluate initial population
        F_vals = np.array([eval_point(x) for x in X])
        best_idx = np.argmin(F_vals)
        best_f = F_vals[best_idx]
        best_x = X[best_idx].copy()

        # DE loop
        for gen in range(n_de // pop_size):
            if time.time() > deadline or calls_made >= budget:
                break

            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                r1, r2, r3 = rng.choice([j for j in range(pop_size) if j != i], size=3, replace=False)

                # Mutation
                v = X[r1] + F * (X[r2] - X[r3])

                # Crossover
                trial = X[i].copy()
                jrand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                f_trial = eval_point(trial)

                # Selection
                if f_trial <= F_vals[i]:
                    X[i] = trial
                    F_vals[i] = f_trial

        # Phase 2: CMA-ES local refinement (remaining 40% of budget)
        n_cma = budget - calls_made
        if n_cma > 10 and best_x is not None:
            # Initialize CMA-ES parameters
            sigma = 0.5  # Initial step size
            pop_size_cma = max(12, dim + int(3 * np.log(dim)))
            lambda_ = pop_size_cma
            mu = lambda_ // 2  # Selection pressure

            # Initialize mean at best point found
            mean = best_x.copy()
            C = np.eye(dim)  # Covariance matrix
            step_size = sigma

            # Track weights for weighted recombination
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / ((dim + 1) ** 2 + mu_eff)
            c_c = (4 - mu_eff + c_sigma) / ((dim + 4) ** 2 + mu_eff)

            # CMA-ES loop
            for step in range(n_cma // lambda_):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                D = np.sqrt(np.diag(C))
                B = np.linalg.cholesky(C)
                B_inv = np.linalg.inv(B)

                # Standard normal samples
                z_samples = rng.standard_normal((lambda_, dim))

                # Transform to search space
                x_samples = mean + np.array([z @ B.T * D for z in z_samples])
                x_samples = np.clip(x_samples, -5, 5)

                # Evaluate
                f_samples = np.array([eval_point(x) for x in x_samples])

                # Sort by fitness
                sorted_idx = np.argsort(f_samples)
                x_selected = x_samples[sorted_idx[:mu]]
                f_selected = f_samples[sorted_idx[:mu]]

                # Update mean
                new_mean = np.sum(weights[:, None] * x_selected, axis=0)

                # Update covariance
                diff = x_selected - mean
                C = (1 - c_c * (1 - c_c)) * C + c_c * (np.outer(new_mean - mean, new_mean - mean) + C)
                C = (C + C.T) / 2  # Symmetrize

                # Update sigma
                sigma *= np.exp(c_sigma * (np.std(np.linalg.norm(diff, axis=1)) / (0.84 + 1.0 / (dim + 1)) - 1))
                sigma = max(sigma, 1e-8)

                # Update mean
                mean = new_mean

                # Ensure positive definite covariance
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_f
