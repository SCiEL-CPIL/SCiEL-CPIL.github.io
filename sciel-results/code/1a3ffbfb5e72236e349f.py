"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with large initial population for rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # Safety margin before 330s kill

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=float).reshape(dim))
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                return 1e10

        # CMA-ES implementation
        # Population size lambda
        lam = 10
        # Number of selected points mu
        mu = lam // 2
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Precompute eigen decomposition for C
        eigvals, eigvecs = np.linalg.eigh(C)
        sqrt_C = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T
        inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T
        
        # Step size control parameters
        d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1)
        c_sigma = (d_sigma + 2) / (dim + d_sigma + 5)
        c_c = (4 + lam / mu - dim) / (dim + 4 + 2 * lam / mu)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((mu + 1) / 2 - 1) / ((dim + 2) ** 2 + lam))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        def update_cma(x_mean, sigma, C, C_inv, p_sigma, p_c, population, values):
            nonlocal eigvals, eigvecs, sqrt_C, inv_sqrt_C
            
            # Sort population by fitness
            sorted_indices = np.argsort(values)
            selected = population[sorted_indices[:mu]]
            selected_weights = weights
            
            # Update mean
            new_mean = np.sum(selected_weights[:, None] * selected, axis=0)
            
            # Compute step
            step = (new_mean - x_mean) / sigma
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_C @ step
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * 1))) / chi_n * (1 + c_c) if np.linalg.norm(p_sigma) > 0 else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * (1 - c_c) - c_c
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            C += c_mu * np.sum(selected_weights[:, None, None] * np.outer(selected - x_mean, selected - x_mean) / (sigma ** 2), axis=0)
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update sigma
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            return new_mean, sigma, C, C_inv, p_sigma, p_c

        # Main CMA-ES loop
        # Initial population: diverse points
        initial_pop = []
        # Start with origin
        initial_pop.append(np.zeros(dim))
        # Add random points in different regions
        for i in range(1, lam):
            if i == 1:
                initial_pop.append(rng.uniform(-4, 4, dim))
            elif i == 2:
                initial_pop.append(rng.uniform(-3, 3, dim))
            else:
                initial_pop.append(rng.uniform(-5, 5, dim))
        
        population = np.array([np.clip(x, -5, 5) for x in initial_pop])
        values = np.array([safe_func(x) for x in population])
        
        # Update best
        best_idx = np.argmin(values)
        best_f = values[best_idx]
        best_x = population[best_idx].copy()
        
        # CMA-ES iterations
        max_iter = 50
        for it in range(max_iter):
            if call_count >= budget or time.time() > deadline:
                break
            
            # Generate population
            population = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * sqrt_C @ z
                y = np.clip(y, -5, 5)
                population.append(y)
            population = np.array(population)
            
            # Evaluate
            values = np.array([safe_func(x) for x in population])
            
            # Update best
            if np.min(values) < best_f:
                best_f = np.min(values)
                best_idx = np.argmin(values)
                best_x = population[best_idx].copy()
            
            # Update CMA-ES parameters
            x_mean, sigma, C, C_inv, p_sigma, p_c = update_cma(
                x_mean, sigma, C, C_inv, p_sigma, p_c, population, values
            )
            
            # If sigma becomes too small, try restarting with larger sigma
            if sigma < 1e-3:
                # Try a few random restarts
                for restart in range(3):
                    if call_count >= budget or time.time() > deadline:
                        break
                    x_restart = rng.uniform(-5, 5, dim)
                    f_restart = safe_func(x_restart)
                    if f_restart < best_f:
                        best_f = f_restart
                        best_x = x_restart.copy()
                        x_mean = x_restart.copy()
                        sigma = 1.5
                        C = np.eye(dim)
                        C_inv = np.eye(dim)
                        eigvals, eigvecs = np.linalg.eigh(C)
                        sqrt_C = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T
                        inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T
                        p_sigma = np.zeros(dim)
                        p_c = np.zeros(dim)
                        break
        
        # Final polishing phase: intensive local search around best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            stagnation = 0
            max_stagnation = 20
            
            while call_count < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, 2.0)
                    stagnation = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    stagnation += 1
                
                if stagnation > max_stagnation:
                    # Try escape moves
                    escaped = False
                    for _ in range(5):
                        if call_count >= budget or time.time() > deadline:
                            break
                        y = np.clip(x + 2.0 * rng.standard_normal(dim), -5, 5)
                        fy = safe_func(y)
                        if fy < fx:
                            x, fx = y, fy
                            sigma = 1.5
                            stagnation = 0
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            escaped = True
                            break
                    if not escaped:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
