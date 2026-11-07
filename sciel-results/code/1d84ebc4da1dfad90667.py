"""Sobol global search followed by 1+1 CMA-ES local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol quasi-random global exploration then 1+1 CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sequence(self, n, dim, seed=42):
        """Generate n points in [0,1)^dim using a Sobol-like quasi-random sequence."""
        # Use a simple Halton-like sequence for quasi-randomness
        # For better coverage, use a scrambled Sobol approach via prime-based Halton
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                p = primes[d % len(primes)]
                f = 1.0
                i_val = i + 1
                while i_val > 0:
                    f /= p
                    i_val, r = divmod(i_val, p)
                    f *= r
                points[i, d] = f
        return points

    def _cma_11(self, func, x0, sigma0, budget, deadline, rng):
        """1+1 CMA-ES with covariance adaptation."""
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Update rates
        mu_sigma = 1.0 / (dim + 4)
        mu_c = (2 + 2.0 / (dim + 1)) / (dim + 4 + 2)
        
        # Eigen decomposition cache
        eigenvalues = np.ones(dim)
        eigenvectors = np.eye(dim)
        eigen_count = 0
        eigen_interval = max(1, dim)
        
        calls = 0
        best_x = x.copy()
        best_fx = fx
        
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Update eigen decomposition periodically
            if eigen_count % eigen_interval == 0:
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigenvalues = np.maximum(eigenvalues, 1e-12)
            
            # Compute step size
            D = 1.0 / np.sqrt(eigenvalues)
            
            # Sample direction
            z = rng.standard_normal(dim)
            y = np.sum(eigenvectors * D[:, np.newaxis], axis=1) * z
            
            # Update path variables
            p_sigma = (1 - mu_sigma) * p_sigma + mu_sigma * (D * y)
            p_c = (1 - mu_c) * p_c + mu_c * y
            
            # Compute h_sigma
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - mu_sigma)**(2 * (calls + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt(np.trace(C_inv @ C) - 1) else 0.0
            
            # Update covariance
            C = (1 + mu_c * (2 - mu_c) - mu_sigma) * C + mu_c * p_c @ p_c + mu_sigma * h_sigma * (D * y) @ (D * y)
            
            # Regularize C
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            # Compute C_inv
            C_inv = np.linalg.inv(C)
            
            # New point
            x_new = x + sigma * np.sum(eigenvectors * D[:, np.newaxis], axis=1) * z
            x_new = np.clip(x_new, -5, 5)
            fx_new = func(x_new)
            calls += 1
            
            if fx_new < fx:
                # Success
                x = x_new
                fx = fx_new
                sigma *= 1.04  # Slight increase on success
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Failure
                sigma *= 0.95  # Slight decrease on failure
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            eigen_count += 1
        
        return best_x, best_fx, calls

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(12345)
        dim = self.dim
        
        # Phase 1: Sobol quasi-random global exploration
        n_global = min(100, self.budget // 5)
        sobol_points = self._sobol_sequence(n_global, dim)
        x_sobol = -5 + 10 * sobol_points  # Map to [-5, 5]^dim
        
        best_x = None
        best_fx = np.inf
        calls = 0
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            x = x_sobol[i]
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Phase 2: 1+1 CMA-ES local refinement from best point
        remaining_budget = self.budget - calls
        if remaining_budget > 0:
            best_x, best_fx, cma_calls = self._cma_11(
                func, best_x, 1.0, remaining_budget, deadline, rng
            )
            calls += cma_calls
        
        # If we still have budget, try a few more local restarts
        if calls < self.budget and best_x is not None:
            n_restarts = min(3, (self.budget - calls) // 50)
            for r in range(n_restarts):
                if time.time() > deadline:
                    break
                # Perturb best point
                x_start = best_x + 0.5 * rng.standard_normal(dim)
                x_start = np.clip(x_start, -5, 5)
                fx_start = func(x_start)
                calls += 1
                if fx_start < best_fx:
                    best_fx = fx_start
                    best_x = x_start.copy()
                
                # Short CMA-ES from this point
                short_budget = min(50, self.budget - calls)
                if short_budget > 0:
                    x_new, fx_new, cma_calls = self._cma_11(
                        func, x_start, 0.5, short_budget, deadline, rng
                    )
                    calls += cma_calls
                    if fx_new < best_fx:
                        best_fx = fx_new
                        best_x = x_new.copy()
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
