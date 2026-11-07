"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive sigma for Schwefel-like functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def evaluate(x):
            nonlocal calls
            fx = func(x)
            calls += 1
            return fx

        # Phase 1: Multi-start to find good initial basins
        # Use 50% of budget for multi-start with (1+1)-ES
        phase1_budget = self.budget // 2
        
        # Try multiple starting points with short (1+1)-ES runs
        num_starts = 8
        calls_per_start = max(10, phase1_budget // num_starts)
        
        for s in range(num_starts):
            if calls >= phase1_budget or time.time() >= deadline:
                break
            
            # Start point: mix of random and structured
            if s == 0:
                x = np.zeros(self.dim)
            elif s == 1:
                x = rng.uniform(-3.0, 3.0, self.dim)
            else:
                x = rng.uniform(-5.0, 5.0, self.dim)
            
            fx = evaluate(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Short (1+1)-ES from this start
            sigma = 1.5
            stall = 0
            max_stall = 15
            
            for _ in range(calls_per_start):
                if calls >= phase1_budget or time.time() >= deadline:
                    break
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    stall = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    stall += 1
                    if stall > max_stall:
                        break
                
                # Simple sigma adaptation
                if stall == 0:
                    sigma *= 1.05
                elif stall > 5:
                    sigma *= 0.9
                sigma = max(0.05, min(sigma, 5.0))

        # Phase 2: CMA-ES from the best point found
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = evaluate(best_x)
        
        if calls >= self.budget or time.time() >= deadline:
            return best_x, best_f
        
        # CMA-ES parameters
        x_mean = best_x.copy()
        sigma = 0.5
        pop_size = 4
        max_generations = (self.budget - calls) // (pop_size + 1)
        
        # CMA-ES matrices
        C = np.eye(self.dim)
        C_inv = np.eye(self.dim)
        C_sqrt = np.eye(self.dim)
        
        # Evolution path
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # CMA-ES parameters
        c_sigma = 1.0 / (pop_size + 3)
        c_c = (4 + pop_size / self.dim) / (pop_size + 4 + 2 * self.dim / self.dim)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Damping
        chi_n = np.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim ** 2))
        
        for gen in range(max_generations):
            if calls >= self.budget or time.time() >= deadline:
                break
            
            # Generate population
            offspring = []
            for i in range(pop_size):
                if calls >= self.budget or time.time() >= deadline:
                    break
                # Use Cholesky-like update (simplified: use C directly for sampling)
                # For simplicity, use diagonal approximation
                z = rng.standard_normal(self.dim)
                # Apply C approximation (use diagonal of C)
                diag_C = np.diag(C)
                y = x_mean + sigma * (diag_C ** 0.5) * z
                y = np.clip(y, -5.0, 5.0)
                offspring.append((y, None))
            
            if not offspring:
                break
            
            # Evaluate offspring
            for i, (y, _) in enumerate(offspring):
                if calls >= self.budget or time.time() >= deadline:
                    break
                fy = evaluate(y)
                offspring[i] = (y, fy)
            
            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            
            # Update mean
            y_w = np.zeros(self.dim)
            for i in range(mu):
                y_w += weights[i] * offspring[i][0]
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ (y_w - x_mean) / sigma)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - x_mean) / sigma
            
            # Update covariance
            delta_h = (np.linalg.norm(p_sigma) ** 2 / chi_n ** 2 > 1 + 4 / self.dim)
            C = (1 + c_c * (2 - c_c) * delta_h - c_c) * C + c_c * (p_c @ p_c.T)
            
            # Update sigma
            sigma *= np.exp(0.5 * c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(0.01, min(sigma, 10.0))
            
            # Update mean
            x_mean = y_w
            
            # Track best
            if offspring[0][1] < best_f:
                best_f = offspring[0][1]
                best_x = offspring[0][0].copy()
            
            # If sigma is very small, we've converged
            if sigma < 0.01:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
