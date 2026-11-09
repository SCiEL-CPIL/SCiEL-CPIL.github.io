"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer with restarts for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        call_count = [0]

        def tracked_func(x):
            call_count[0] += 1
            return func(x)

        # Phase 1: Quick random sampling for early progress
        n_init = min(10, budget)
        rng = np.random.default_rng(42)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = tracked_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        remaining = budget - call_count[0]
        if remaining <= 0:
            return best_x, best_f

        # Phase 2: CMA-ES with restarts
        # Use multiple restarts to handle multimodality
        # Allocate budget across restarts
        n_restarts = 3
        budget_per_restart = max(10, remaining // n_restarts)

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if call_count[0] >= budget:
                break

            # Starting point for this restart
            if r == 0:
                x0 = best_x.copy()
            else:
                # Perturb best or random restart
                x0 = best_x + 2.0 * rng.standard_normal(dim)
                x0 = np.clip(x0, -5, 5)

            # Run CMA-ES via scipy minimize
            # Use Nelder-Mead as fallback if CMA-ES not available, but let's try
            # Actually, let's implement a simple CMA-ES manually for more control
            # Or use scipy's cmaes if available

            # Let's use a manual CMA-ES implementation for better control
            cmaes_result = self._cmaes(tracked_func, x0, dim, budget_per_restart,
                                       deadline, call_count, best_f, best_x)
            if cmaes_result[1] < best_f:
                best_f = cmaes_result[1]
                best_x = cmaes_result[0]

        return best_x, best_f

    def _cmaes(self, func, x0, dim, budget, deadline, call_count, best_f, best_x):
        """Simple CMA-ES implementation."""
        import numpy as np

        # CMA-ES parameters
        pop_size = max(4 + int(3 * np.log(dim)), 6)
        pop_size = min(pop_size, budget // 2)  # ensure we can do at least 2 generations
        pop_size = max(pop_size, 4)

        # Initial sigma
        sigma = 1.0

        # CMA-ES state
        mean = x0.copy().astype(float)
        C = np.eye(dim)
        C_inv = np.eye(dim)

        # Step size control
        cs = (pop_size + 2) / (dim + pop_size + 2)
        ds = 1 + 2 * max(0, (np.sqrt((pop_size + 1) / (dim + 1)) - 1) / cs) + cs
        cm = 2 / ((dim + 1.3) ** 2 + pop_size)

        # Evolution path
        ps = np.zeros(dim)

        # Precompute
        def sqrtm(matrix):
            """Matrix square root via eigendecomposition."""
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

        def invsqrtm(matrix):
            """Inverse matrix square root."""
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        # Cholesky of C
        try:
            B = np.linalg.cholesky(C)
            D = np.diag(B)
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            B = np.eye(dim)
            D = np.ones(dim)
            B_inv = np.eye(dim)

        rng = np.random.default_rng(123)

        while call_count[0] < budget:
            if time.time() > deadline:
                break

            # Generate population
            samples = []
            fvals = []
            for i in range(pop_size):
                if call_count[0] >= budget:
                    break
                z = rng.standard_normal(dim)
                # Transform: x = mean + sigma * B @ z
                x = mean + sigma * (B_inv.T @ z)  # B_inv.T gives B^{-T}, but we want B @ z
                # Actually, x = mean + sigma * C^{1/2} @ z = mean + sigma * B.T @ z
                # Let me redo: if C = B @ B.T, then C^{1/2} = B.T (not quite)
                # Standard: sample y = B @ z where B is such that B @ B.T = C
                # Using Cholesky: C = L @ L.T, so y = L.T @ z
                x = mean + sigma * (B.T @ z)
                x = np.clip(x, -5, 5)
                fx = func(x)
                samples.append(x)
                fvals.append(fx)

            if len(samples) < 2:
                break

            samples = np.array(samples)
            fvals = np.array(fvals)

            # Sort by fitness
            order = np.argsort(fvals)
            samples = samples[order]
            fvals = fvals[order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = samples[0].copy()

            # Weighted recombination
            weights = np.exp(-np.arange(len(fvals)) / (len(fvals) / 2.0))
            weights /= weights.sum()
            mu = int(np.sum(weights > 0))  # effective number of parents

            # Update mean
            new_mean = np.sum(weights[:mu, np.newaxis] * samples[:mu], axis=0)

            # Update evolution path
            C_inv_sqrt = invsqrtm(C)
            ps = (1 - cs) * ps + cs / np.sqrt(cs * (2 - cs)) * C_inv_sqrt @ (new_mean - mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2)) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 2)

            # Update covariance
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            delta_sig = 1 + 2 * max(0, min((np.linalg.norm(ps) ** 2 / dim - 1) / (1 - (1 - cs) ** 2), 2)) * cs

            C = (1 + cm * delta_hsig - cm) * C + cm * (np.outer(ps, ps) / (1 - (1 - cs) ** 2) +
                                                         np.sum(weights[:mu, np.newaxis] *
                                                                np.outer(C_inv_sqrt @ (samples[:mu] - mean).T / sigma,
                                                                        C_inv_sqrt @ (samples[:mu] - mean).T / sigma),
                                                                axis=0) / sigma ** 2)

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-10 * np.eye(dim)

            # Update sigma
            sigma *= np.exp((ds / (5 * dim)) * (np.linalg.norm(C_inv_sqrt @ (new_mean - mean)) / sigma - 1))
            sigma = np.clip(sigma, 1e-6, 100)

            # Update Cholesky
            try:
                B = np.linalg.cholesky(C)
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                # Reset if not positive definite
                C = np.eye(dim)
                B = np.eye(dim)
                B_inv = np.eye(dim)

            mean = new_mean

            # If progress is slow, break
            if len(fvals) > 0 and fvals[0] - fvals[-1] < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
