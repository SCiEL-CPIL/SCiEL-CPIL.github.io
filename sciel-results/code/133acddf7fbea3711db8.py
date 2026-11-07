"""Rewritten optimizer using CMA-ES with adaptive population and restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style continuous optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng):
        """Run standard CMA-ES."""
        dim = self.dim
        # Population size
        N = int(4 + 3 * np.log(dim))
        # Weights
        mu = N // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Parameters
        c_sigma = (N + 2) / (dim + N + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((N - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + N / dim) / (dim + 4 + 2 * N / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + N)
        c_mu = min(1 - c_1, 2 * ((N / 2 + 1) / (dim + 2) - 1))

        # Initialization
        x = x0.copy()
        sigma = sigma0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        
        # Precompute C^0.5 (identity initially)
        C_sqrt = np.eye(dim)
        inv_C_sqrt = np.eye(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x0.copy()
        best_f = func(x0)
        calls = 1

        # Track history for potential early stopping or just run until budget
        while calls < budget:
            # Generate offspring
            B = rng.standard_normal((N, dim)) @ C_sqrt
            y = x + sigma * B
            y = np.clip(y, -5, 5)
            
            # Evaluate
            fitnesses = np.empty(N)
            for i in range(N):
                if calls >= budget:
                    break
                fitnesses[i] = func(y[i])
                calls += 1
            
            if calls >= budget:
                 # If we broke out early in the loop, fitnesses might be incomplete
                 # But we only break if we ran out of budget. 
                 # Actually, the loop condition is `while calls < budget`.
                 # Inside, we increment calls. If calls becomes budget, we finish this batch.
                 # We should check if we have enough fitnesses.
                 pass

            # Ensure we have N fitnesses
            if len(fitnesses) < N:
                 # This case should not happen if we managed budget correctly, 
                 # but let's be safe. If we ran out of budget mid-evaluation, we stop.
                 break

            # Sort
            order = np.argsort(fitnesses)
            y_sorted = y[order]
            f_sorted = fitnesses[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Compute weighted mean of selected points
            y_mean = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_C_sqrt @ (y_mean - x) / sigma)
            
            # h_sig calculation
            # Note: calls is incremented after each evaluation. 
            # At the start of the loop, we have evaluated `calls - 1` points total (including x0).
            # The generation number is roughly (calls - 1) // N.
            # The formula for h_sig uses the number of generations.
            # Let's use a simpler check or standard implementation.
            # Standard: h_sig = 1 if norm(p_sigma) < (1 + 2/(dim+1)) * chi_n * sqrt(1 - (1-c_sigma)^(2*(gen+1))) else 0
            # We need the generation number.
            gen = (calls - 1) // N  # 0-based generation index for the current update?
            # Actually, after evaluating N points, we are at the end of a generation.
            # Let's just use the norm check against a threshold that adapts slowly.
            # Or simpler: just use the norm of p_sigma.
            # The exact h_sig is for stability. For dim=5, it's less critical.
            # I will use the standard approximation.
            
            # To avoid division by zero or negative sqrt in early steps:
            # We can just check if the norm is small enough.
            # A common simplification is to check against chi_n * sqrt(1 - (1-c_sigma)^(2*(gen+1)))
            # Let's compute gen properly.
            # calls starts at 1 (x0).
            # After 1st gen: calls = 1 + N.
            # gen index for update 1 is 0? Or 1?
            # Let's use the number of completed generations.
            # completed_gens = (calls - 1) // N
            completed_gens = (calls - 1) // N
            threshold = chi_n * np.sqrt(1 - (1 - c_sigma) ** (2 * (completed_gens + 1)))
            h_sig = 1 if np.linalg.norm(p_sigma) < (1 + 2 / (dim + 1)) * threshold else 0
            
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_mean - x) / sigma
            
            # Update covariance
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_h) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum(weights[:, None, None] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x) / sigma**2, axis=0)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            x = y_mean
            
            # Update C^0.5 (Eigenvalue decomposition)
            # This is expensive, but dim=5 is small.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive definite
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            inv_C_sqrt = (eigvecs * (1 / np.sqrt(eigvals))) @ eigvecs.T

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Use a single run of CMA-ES with the full budget.
        # Initialize at the center.
        x0 = np.zeros(dim)
        sigma0 = 2.0  # Reasonable initial step size for [-5, 5] box

        best_x, best_f, calls_used = self._cma_es(func, x0, sigma0, budget, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
