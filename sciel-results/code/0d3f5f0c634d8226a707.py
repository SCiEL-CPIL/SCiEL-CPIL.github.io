"""Rewritten optimizer: CMA-ES with adaptive restarts and diverse initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES based optimizer with adaptive restarts for multimodal functions.
    
    Uses a population-based CMA-ES strategy that adapts both position and step
    size covariance. Multiple restarts from diverse starting points help escape
    local minima on Rastrigin and Gallagher functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_exceeded():
            return time.time() > deadline

        # ---------------------------------------------------------
        # Phase 1: Diverse Initialization
        # ---------------------------------------------------------
        # Evaluate a diverse set of points to find good starting regions
        
        # High priority points
        high_priority = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
        ]
        
        # Integer lattice samples for Rastrigin
        lattice_vals = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        lattice_samples = []
        for _ in range(20):
            pt = np.array([rng.choice(lattice_vals) for _ in range(dim)])
            lattice_samples.append(pt)
        
        # Random uniform samples
        random_samples = [rng.uniform(-5, 5, dim) for _ in range(30)]
        
        # Combine all initialization candidates
        init_candidates = high_priority + lattice_samples + random_samples
        
        for pt in init_candidates:
            if calls >= budget or time_exceeded():
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # ---------------------------------------------------------
        # Phase 2: CMA-ES with Adaptive Restarts
        # ---------------------------------------------------------
        
        # CMA-ES parameters
        lambda_pop = max(4, int(4 + 2 * np.log(dim)))  # Population size
        mu = lambda_pop // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Restart strategy
        max_restarts = 8
        restart_budget = max(30, (budget - calls) // max_restarts)
        
        for restart in range(max_restarts):
            if calls >= budget or time_exceeded():
                break
            
            # Choose starting point for this restart
            if restart == 0:
                x = best_x.copy()
                sigma = 1.0
            else:
                # Mix of global and local restarts
                if rng.random() < 0.3:
                    x = rng.uniform(-5, 5, dim)
                    sigma = 2.0
                else:
                    x = np.clip(best_x + rng.normal(0, 1.0, dim), -5, 5)
                    sigma = 0.5
            
            fx = func(x)
            calls += 1
            update_best(x, fx)
            
            # CMA-ES state
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c1 = (2 / ((dim + 1.) ** 2 + mu_eff)) / (dim * (dim + 1.))
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2.) ** 2 + mu_eff))
            chiN = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1 / (21 * dim ** 2))
            
            steps_in_restart = 0
            max_steps = restart_budget
            stagnation = 0
            best_f_restart = fx
            
            for step in range(max_steps):
                if calls + lambda_pop > budget or time_exceeded():
                    break
                
                # Generate population
                population = []
                for i in range(lambda_pop):
                    y = x + sigma * (C_sqrt @ rng.standard_normal(dim))
                    y = np.clip(y, -5, 5)
                    population.append(y)
                
                # Evaluate population
                fitness = []
                for y in population:
                    fy = func(y)
                    calls += 1
                    fitness.append(fy)
                    update_best(y, fy)
                
                # Sort by fitness
                indices = np.argsort(fitness)
                population = [population[i] for i in indices]
                fitness = [fitness[i] for i in indices]
                
                # Update mean
                x_old = x.copy()
                x = sum(weights[i] * population[i] for i in range(mu))
                x = np.clip(x, -5, 5)
                
                # Update step size and covariance
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C_sqrt.T @ (x - x_old))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (step + 1))) < (chiN + 1.4) * np.sqrt(1 + cs):
                    pc = (1 - c1) * pc + c1 / sigma * (x - x_old)
                    C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * sum(
                        weights[i] * np.outer(population[i] - x_old, population[i] - x_old)
                        for i in range(mu)
                    )
                
                # Update C_sqrt via Cholesky
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = (C + C.T) / 2
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-10)
                    C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                    C_sqrt = np.linalg.cholesky(C)
                
                # Adapt sigma
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 0.01, 10.0)
                
                # Check for improvement
                if fitness[0] < best_f_restart - 1e-8:
                    best_f_restart = fitness[0]
                    stagnation = 0
                else:
                    stagnation += 1
                
                steps_in_restart += 1
                
                # Early stop if converged
                if sigma < 0.05 and stagnation > 10:
                    break
            
            # If we've used most of the budget, stop restarting
            if calls > budget * 0.85:
                break
        
        # ---------------------------------------------------------
        # Phase 3: Final Polish with (1+1)-ES
        # ---------------------------------------------------------
        if calls < budget and not time_exceeded():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            
            while calls < budget and not time_exceeded():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                sigma = np.clip(sigma, 0.001, 1.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
