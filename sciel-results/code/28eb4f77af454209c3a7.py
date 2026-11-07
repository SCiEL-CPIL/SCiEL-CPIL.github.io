"""Optimizer for BBOB-style Bent Cigar (f12) with CMA-ES + L-BFGS-B refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    """CMA-ES with L-BFGS-B refinement for Bent Cigar (f12)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin out of 300s

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Wrap function to count calls
        call_count = [0]

        def wrapped(x):
            call_count[0] += 1
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x[:dim] if x.size >= dim else np.pad(x, (0, dim - x.size))
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            return float(val)

        # Phase 1: CMA-ES with adaptive sigma
        # Initial point: try a few random starts, pick the best
        n_starts = min(10, budget // 10)
        best_x = None
        best_f = np.inf
        for _ in range(n_starts):
            x0 = rng.uniform(-4.0, 4.0, dim)
            f0 = wrapped(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # CMA-ES parameters
        sigma = 1.5
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        # Success rate adaptation
        successes = 0
        window = pop_size

        x_mean = best_x.copy()
        f_mean = best_f
        best_x_es = best_x.copy()
        best_f_es = best_f

        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # CMA-ES hyperparameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Inverse square root of C for step size
        def invsqrtm(M):
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals = np.maximum(eigvals, 1e-20)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        D = invsqrtm(C)

        iterations = 0
        while call_count[0] < budget:
            # Check time
            if time.time() > deadline - 60:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (D @ z)
                y = np.clip(y, -5.0, 5.0)
                fy = wrapped(y)
                offspring.append((fy, y))
                if call_count[0] >= budget:
                    break

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            for fy, y in offspring:
                if fy < best_f_es:
                    best_f_es = fy
                    best_x_es = y.copy()
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Compute weighted mean
            x_new_mean = np.zeros(dim)
            for i in range(min(mu, len(offspring))):
                x_new_mean += weights[i] * offspring[i][1]

            # Update evolution paths
            # Need to transform the weighted mean difference back to original space
            diff = (x_new_mean - x_mean) / sigma
            # Transform through D^{-1}
            D_inv = invsqrtm(C)
            diff_orig = D_inv @ diff

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff_orig
            chi_n = np.sqrt(dim) * (1 + 1 / (2 * dim) + 1 / (8 * dim ** 2))

            # Update covariance
            delta_h = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iterations + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            p_c = (1 - c_c) * p_c + delta_h * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff_orig

            # Update C
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(min(mu, len(offspring))):
                zi = (offspring[i][1] - x_mean) / sigma
                zi_orig = D_inv @ zi
                rank_mu += weights[i] * np.outer(zi_orig, zi_orig)

            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Adapt sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            x_mean = x_new_mean
            D = invsqrtm(C)
            iterations += 1

            # Check if we should stop CMA-ES and move to refinement
            if best_f_es < 1e-3 and call_count[0] > budget // 2:
                break

        # Phase 2: L-BFGS-B refinement from the best point found
        if call_count[0] < budget - 5:
            try:
                result = scipy_minimize(
                    lambda x: wrapped(x),
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfun': max(1, budget - call_count[0] - 1), 'gtol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass

        # Final check: make sure we haven't exceeded budget
        # Return the best point found
        return best_x, best_f
# EVOLVE-BLOCK-END
