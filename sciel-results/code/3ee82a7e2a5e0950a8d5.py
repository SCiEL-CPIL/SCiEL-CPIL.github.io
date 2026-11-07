"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).

Uses a corrected, efficient CMA-ES with proper stagnation detection
and adaptive restarts. Structural change: simplified covariance update
without full eigendecomposition per step, better budget allocation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Corrected CMA-ES with adaptive restarts for ill-conditioned problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_run(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        Corrected CMA-ES implementation.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        # Population size: n = 4 + 3*ln(dim) for dim=5 -> ~7, use 6 for speed
        n = max(4, int(4 + 3 * np.log(dim)))
        mu = max(1, n // 2)
        
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(dim)
        calls = 0
        
        # CMA-ES parameters (standard values)
        c_sigma = (1 + 2 * max(0, np.sqrt((n - 1) / dim) - 1)) / (dim + 2)
        c_c = (4 + n / dim) / (dim + 4 + 2 * n / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + n)
        c_mu = min(1 - c_1, 2 * (2 - (dim + 2) / (n + 4)) / ((dim + 2) ** 2 + n))
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Evaluate initial point
        best_f = func(x)
        best_x = x.copy()
        calls = 1
        
        # Track stagnation
        best_f_prev = best_f
        no_improve_count = 0
        stagnation_window = 5  # generations without improvement triggers restart consideration
        
        while calls < max_calls and time.time() < deadline:
            # Generate population
            Z = rng.standard_normal((n, dim))
            Y = x[None, :] + sigma * (Z @ C.T)  # Y[i] = x + sigma * C @ Z[i].T
            # Actually need: Y[i] = x + sigma * (L @ Z[i]) where C = L@L.T
            # Let's use: Y = x + sigma * (Z @ C) if we define C properly
            # Standard: Y_i = x + sigma * B @ Z_i where B B^T = C
            # Simpler: use C directly as covariance, sample via Cholesky
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            Y = x[None, :] + sigma * (Z @ L.T)
            
            f_list = []
            for i in range(n):
                if calls >= max_calls or time.time() > deadline:
                    break
                y = Y[i]
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                f_list.append(fy)
            
            if len(f_list) < 2:
                break
            
            # Sort by fitness
            idx = np.argsort(f_list)
            Y_sorted = Y[idx]
            f_sorted = np.array(f_list)[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = Y_sorted[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Check for stagnation - restart with larger sigma if stuck
            if no_improve_count >= stagnation_window:
                # Perturb center and increase sigma
                x = best_x + rng.standard_normal(dim) * sigma * 0.5
                sigma *= 1.5
                no_improve_count = 0
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                C = np.eye(dim)
                continue
            
            # Recombination
            Y_w = np.zeros(dim)
            for i in range(min(mu, len(Y_sorted))):
                Y_w += weights[i] * Y_sorted[i]
            
            x_old = x.copy()
            x = Y_w
            
            # Compute covariance update
            D = (Y_sorted[:mu] - x_old[None, :]) / sigma
            C_update = np.zeros((dim, dim))
            for i in range(mu):
                C_update += weights[i] * np.outer(D[i], D[i])
            
            # Path updates
            h_sigma = np.sqrt((1 - np.exp(-2 * (calls + 1) * c_sigma)) * (1 - np.exp(-2 * c_sigma)))
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ ((x - x_old) / sigma))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * ((x - x_old) / sigma)
            
            # Update C
            delta_h = (1 - np.exp(-2 * (calls + 1) * c_c)) * (1 - np.exp(-2 * c_c))
            C = (1 + c_1 * (mu_eff - 1) - c_1 - c_mu) * C \
                + c_1 / sigma ** 2 * np.outer(x - x_old, x - x_old) \
                + c_mu / sigma ** 2 * C_update
            
            # Ensure PSD and symmetric
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Convergence check
            if sigma < 1e-8:
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Efficient initial sampling - use fewer but more spread out points
        n_samples = min(8, budget // 10)
        # Use Latin hypercube-like sampling for better coverage
        for i in range(n_samples):
            if total_calls >= budget or time.time() > deadline - 5.0:
                break
            x = rng.uniform(-4, 4, size=dim)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES with adaptive restarts
        if total_calls < budget:
            remaining = budget - total_calls
            
            # Run CMA-ES from best point found
            x0 = best_x.copy()
            sigma0 = 1.0
            
            x_best, f_best, calls_used = self._cma_es_run(
                func, rng, x0, sigma0, remaining, deadline
            )
            total_calls += calls_used
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
        
        # Phase 3: If we have budget left, do a final local refinement
        remaining = budget - total_calls
        if remaining >= 5 and time.time() < deadline - 5.0:
            x_local = best_x.copy()
            sigma_local = max(0.01, 0.1 * np.std(best_x))
            
            x_polish, f_polish, calls_polish = self._cma_es_run(
                func, rng, x_local, sigma_local, remaining, deadline
            )
            total_calls += calls_polish
            
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
