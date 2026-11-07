"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).

Uses restartable CMA-ES with Cholesky-based sampling for efficiency.
Structural change: single CMA-ES run with stagnation-triggered restarts,
faster per-generation computation via Cholesky decomposition.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Restartable CMA-ES for ill-conditioned BBOB-style problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_single(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        CMA-ES with Cholesky-based sampling.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        n = max(4, int(4 + 3 * np.log(dim)))
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(dim)
        L = np.eye(dim)  # Cholesky factor: C = L @ L.T
        calls = 0
        
        # CMA-ES parameters
        c_sigma = (1 + 2 * max(0, np.sqrt((n - 1) / dim) - 1)) / (dim + 2)
        c_c = (4 + n / dim) / (dim + 4 + 2 * n / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + n)
        c_mu = min(1 - c_1, 2 * (2 - (dim + 2) / (n + 4)) / ((dim + 2) ** 2 + n))
        mu = n // 2
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
        
        gen = 0
        while calls < max_calls and time.time() < deadline:
            gen += 1
            # Generate population using Cholesky
            y_list = []
            f_list = []
            for i in range(n):
                if calls >= max_calls or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                y_list.append(y)
                f_list.append(fy)
            
            if len(y_list) < 2:
                break
            
            # Sort by fitness
            idx = np.argsort(f_list)
            y_sorted = np.array([y_list[i] for i in idx])
            f_sorted = [f_list[i] for i in idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()
            
            # Recombination
            y_w = np.zeros(dim)
            for i in range(min(mu, len(y_sorted))):
                y_w += weights[i] * y_sorted[i]
            
            x_old = x.copy()
            x = y_w
            
            # Compute C update
            C_update = np.zeros((dim, dim))
            for i in range(min(mu, len(y_sorted))):
                d = (y_sorted[i] - x_old) / sigma
                C_update += weights[i] * np.outer(d, d)
            
            h_sigma = np.sqrt((1 - np.exp(-2 / (calls * c_sigma))) / (1 - np.exp(-2 / (calls * c_sigma))))
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (x - x_old))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x - x_old)
            
            # Update C
            C = (1 + c_1 * (mu_eff - 1) - c_1 - c_mu) * C \
                + c_1 / sigma ** 2 * np.outer(x - x_old, x - x_old) \
                + c_mu / sigma ** 2 * C_update
            
            # Ensure PSD
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update Cholesky factor
            L = np.linalg.cholesky(C)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)
            
            # Convergence check
            if sigma < 1e-10:
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
        
        # Phase 1: Initial random sampling
        n_samples = min(15, budget // 8)
        samples = rng.uniform(-4, 4, size=(n_samples, dim))
        sample_values = []
        for i in range(n_samples):
            if total_calls >= budget or time.time() > deadline - 5.0:
                break
            x = samples[i]
            fx = func(x)
            total_calls += 1
            sample_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES with restarts
        if total_calls < budget:
            sorted_idx = np.argsort(sample_values)
            remaining_budget = budget - total_calls
            
            # Use multiple starting points with restart logic
            n_starts = min(3, len(sorted_idx))
            
            for start_i in range(n_starts):
                if total_calls >= budget or time.time() > deadline - 5.0:
                    break
                
                x0 = samples[sorted_idx[start_i]].copy()
                sigma0 = 1.0
                
                # Budget allocation: give more budget to earlier starts
                if start_i == 0:
                    alloc = int(remaining_budget * 0.5)
                elif start_i == 1:
                    alloc = int(remaining_budget * 0.3)
                else:
                    alloc = remaining_budget - total_calls - (budget - total_calls)
                    alloc = max(20, alloc)
                
                x_best, f_best, calls_used = self._cma_es_single(
                    func, rng, x0, sigma0, alloc, deadline
                )
                total_calls += calls_used
                
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()
        
        # Phase 3: Final polish
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline - 5.0:
            x_local = best_x.copy()
            sigma_local = 0.05
            
            x_polish, f_polish, calls_polish = self._cma_es_single(
                func, rng, x_local, sigma_local, remaining, deadline
            )
            total_calls += calls_polish
            
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
