"""Rewritten optimizer using CMA-ES for black-box continuous optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box continuous optimization on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin before 330s kill

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(d))
        lam = 4 + int(3 * np.log(d)) if d > 1 else 4
        # Ensure at least 4
        lam = max(lam, 4)

        # Initial center: center of the box
        mean = np.zeros(d)
        # Initial sigma: start moderately large to explore
        sigma = 1.0

        # Covariance matrix
        C = np.eye(d)

        # Step size control (CSA - Covariance Matrix Adaptation with step-size adaptation)
        cc = (4 + lam / d) / (d + 4 + 2 * lam / d)
        cs = (lam / 2 + 1 + 2 / (d - 1)) / (d + lam / 2 + 5)
        c1 = 2 / ((d + 1.3) ** 2 + lam)
        cmu = min(1 - c1, 2 * ((lam / 2 - 1 + 1 / d) / (d + 2) ** 2 - c1))
        cmu = max(cmu, 0)

        # Step size control parameters
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Path variables
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Track best
        best_x = mean.copy()
        best_f = np.inf

        # Initial evaluation
        best_f = func(best_x)

        # Number of evaluations used
        n_evals = 1

        # Precompute weights
        mu = lam // 2  # number of selected individuals
        weights = np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Generate initial population
        def generate_population():
            """Generate a population of candidates."""
            z = rng.standard_normal((lam, d))
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                L = np.eye(d)
            candidates = mean[None, :] + sigma * (z @ L.T)
            # Clip to box
            candidates = np.clip(candidates, -5, 5)
            return candidates

        # Main loop
        while n_evals < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            candidates = generate_population()

            # Evaluate all candidates
            fitnesses = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    # Fill remaining with a large value
                    fitnesses[i:] = np.inf
                    break
                try:
                    fitnesses[i] = func(candidates[i])
                except Exception:
                    fitnesses[i] = np.inf
                n_evals += 1

            # Sort by fitness
            order = np.argsort(fitnesses)
            candidates_sorted = candidates[order]
            fitnesses_sorted = fitnesses[order]

            # Update best
            if fitnesses_sorted[0] < best_f:
                best_f = fitnesses_sorted[0]
                best_x = candidates_sorted[0].copy()

            # Select top mu individuals
            selected = candidates_sorted[:mu]
            weights_norm = weights / weights.sum()

            # New mean
            new_mean = np.sum(weights_norm[:, None] * selected, axis=0)

            # Update step size control
            y = (new_mean - mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (n_evals / (cs * (d + 1))))) <= (1.4 + 2 / (d + 1)) * chi_n

            # Update covariance matrix adaptation
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y
            delta_hsig = (1 - hsig) * cc * (2 - cc)

            # Update C
            C = (1 + c1 * (mu_eff - 2 + delta_hsig) - c1 - cmu) * C
            C += c1 * np.outer(p_c, p_c)
            C += cmu * np.sum(weights_norm[:, None, None] * np.outer(selected - mean, selected - mean), axis=0) / sigma ** 2

            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update mean
            mean = new_mean

            # Clip mean to box (soft constraint)
            mean = np.clip(mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
