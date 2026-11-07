"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive step size for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0

        def check_time():
            return time.time() > deadline

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Multi-start: try several initial points with different strategies
        # Rastrigin has many local minima, so we need good global exploration
        # Strategy: combine random restarts with a CMA-ES-like local search

        # First, do a quick random scan to find a good basin
        scan_points = 50
        scan_size = 10
        for s in range(scan_points):
            if calls >= budget or check_time():
                break
            # Generate a cluster of points around a random center
            center = rng.uniform(-4.5, 4.5, dim)
            for j in range(scan_size):
                if calls >= budget or check_time():
                    break
                x = center + rng.normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)
                f = func(x)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        if calls >= budget or check_time():
            return best_x.copy(), best_f

        # Now do CMA-ES-like local search around the best point
        # Use a population-based approach with adaptive step size
        pop_size = 8
        lambda_ = pop_size
        mu = pop_size // 2  # number of parents
        
        # Initialize population around best point
        sigma = 1.0
        mean = best_x.copy()
        
        # Create initial population
        population = np.zeros((pop_size, dim))
        for i in range(pop_size):
            population[i] = mean + sigma * rng.standard_normal(dim)
            population[i] = np.clip(population[i], -5, 5)
        
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            fitness[i] = func(population[i])
            calls += 1
            if fitness[i] < best_f:
                best_f = fitness[i]
                best_x = population[i].copy()
        
        # CMA-ES main loop
        weight = np.zeros(mu)
        total_weight = 0
        for i in range(mu):
            weight[i] = np.log(mu + 0.5) - np.log(i + 1)
        total_weight = np.sum(weight)
        weight /= total_weight
        mu_eff = 1.0 / np.sum(weight ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        stagnation = 0
        max_stagnation = 50
        
        while calls < budget and not check_time():
            # Generate offspring
            population = np.zeros((pop_size, dim))
            for i in range(pop_size):
                # Covariance matrix adaptation
                B, D = np.linalg.eigh(C)
                y = rng.standard_normal(dim)
                z = B @ (D * y)
                population[i] = mean + sigma * z
                population[i] = np.clip(population[i], -5, 5)
            
            # Evaluate offspring
            fitness = np.zeros(pop_size)
            for i in range(pop_size):
                if calls >= budget or check_time():
                    break
                fitness[i] = func(population[i])
                calls += 1
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = population[i].copy()
            
            if calls >= budget or check_time():
                break
            
            # Sort by fitness
            order = np.argsort(fitness)
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weight[i] * (population[order[i]] - mean) / sigma
            
            # Update mean
            mean = mean + np.sum(weight * population[order[:mu]], axis=0)
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * y_w
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update evolution path for covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update covariance matrix
            delta_heav = (1 + 2 * max(0, (np.linalg.norm(pc) ** 2 / chi_n - 1) / (dim + 1))) * c_c
            C = (1 - c_1 - c_mu + delta_heav) * C
            C += c_1 * (np.outer(pc, pc) + (delta_heav * chi_n ** 2 - 1) * C)
            
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += weight[i] * np.outer(population[order[i]] - mean, population[order[i]] - mean)
            C += c_mu * H
            
            # Check for stagnation
            if np.all(fitness[order[:mu]] > best_f + 1e-6):
                stagnation += 1
            else:
                stagnation = 0
            
            if stagnation > max_stagnation:
                # Restart with a new random point
                stagnation = 0
                mean = rng.uniform(-4.5, 4.5, dim)
                sigma = 1.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
        
        return best_x.copy(), best_f
# EVOLVE-BLOCK-END
