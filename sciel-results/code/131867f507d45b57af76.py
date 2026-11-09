"""CMA-ES with adaptive step size and local polish for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and local polish for multimodal Weierstrass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            return x, fx

        def check_time():
            return time.time() < deadline and calls_made < budget

        # Phase 1: CMA-ES
        # Initialize
        x = rng.uniform(-5, 5, dim)
        x, fx = call_func(x)
        best_x, best_f = x.copy(), fx
        sigma = 2.0

        # CMA-ES parameters
        pop_size = 4 + int(np.floor(3 * np.log(dim)))
        pop_size = max(pop_size, 4)
        
        # Covariance matrix
        C = np.eye(dim)
        step_size = np.ones(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Parameters
        c_sigma = (pop_size + 2) / (dim + 4)
        c_c = (4 + dim / pop_size) / (dim + 4 + 2)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + pop_size))

        # Track best
        archive = [(fx, x.copy())]

        while check_time():
            # Generate population
            population = []
            fitnesses = []
            
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = x + sigma * np.dot(np.linalg.cholesky(C) @ z, step_size)
                y, fy = call_func(y)
                population.append(y)
                fitnesses.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    archive.append((fy, y.copy()))

            # Sort by fitness
            sorted_idx = np.argsort(fitnesses)
            sorted_fitnesses = np.array(fitnesses)[sorted_idx]
            sorted_population = [population[i] for i in sorted_idx]

            # Update best
            if sorted_fitnesses[0] < best_f:
                best_f = sorted_fitnesses[0]
                best_x = sorted_population[0].copy()

            # Mean update
            x_old = x.copy()
            x_new = sum(weights[i] * sorted_population[i] for i in range(mu))
            
            # Step size update (1/5 rule variant)
            diff = x_new - x_old
            sigma *= np.exp(0.3 * (np.linalg.norm(diff) / (sigma * np.sqrt(mu_eff)) - 1.0))
            sigma = max(1e-8, min(sigma, 10.0))

            # Evolution path updates
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff / (sigma * np.linalg.norm(step_size))
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / (sigma * np.linalg.norm(step_size))
            
            # Covariance update
            v = p_c / np.linalg.norm(p_c) if np.linalg.norm(p_c) > 0 else np.zeros(dim)
            C = (1 + c_1 * (mu_eff - 1) - c_1) * C + c_1 * np.outer(v, v)
            for i in range(mu):
                y_i = (sorted_population[i] - x_old) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            x = x_new

            # Check for stagnation
            if len(archive) > 10:
                recent = archive[-10:]
                if all(f >= best_f + 1e-6 for f, _ in recent[1:]):
                    # Stagnation detected, try local restart
                    break

        # Phase 2: Local (1+1)-ES polish around best
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        stagnation_count = 0
        
        while check_time():
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            y, fy = call_func(y)
            if fy < fx:
                x, fx = y, fy
                best_f = fx
                best_x = y.copy()
                sigma *= 1.2
                stagnation_count = 0
            else:
                sigma *= 0.85
                stagnation_count += 1
                if stagnation_count > 30:
                    # Try a different local neighborhood
                    x = best_x + 0.5 * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    x, fx = call_func(x)
                    sigma = 0.3
                    stagnation_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
