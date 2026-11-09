"""Optimizer for BBOB f23 (Katsuura) using Derandomized Evolutionary Strategy (DES)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Derandomized Evolutionary Strategy (DES) for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget
        bounds = np.array([-5.0] * dim)
        upper = np.array([5.0] * dim)
        
        calls_made = 0
        x_global_best = None
        fx_global_best = np.inf

        def update_best(x, fx):
            nonlocal x_global_best, fx_global_best
            if fx < fx_global_best:
                fx_global_best = fx
                x_global_best = x.copy()

        # Phase 1: Latin Hypercube Sampling for global coverage
        n_lhs = 30
        rng = np.random.default_rng(42)
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, i] = (perm + 0.5) / n_lhs * 10.0 - 5.0

        for i in range(n_lhs):
            if calls_made >= budget:
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            if time.time() > deadline:
                return x_global_best, fx_global_best

        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x_global_best, fx_global_best
            
        # Phase 2: Derandomized Evolutionary Strategy (DES)
        # Use a fixed population of deterministic mutation vectors
        # This provides systematic exploration without random restarts
        
        # Population size for DES
        pop_size = 20
        # Pre-compute deterministic mutation directions
        # Use a set of orthogonal-ish directions to cover the space well
        mutation_vectors = np.zeros((pop_size, dim))
        # First dim vectors are unit vectors
        for i in range(min(dim, pop_size)):
            mutation_vectors[i, i] = 1.0
        # Add some diagonal and mixed directions
        for i in range(dim, pop_size):
            mutation_vectors[i] = rng.standard_normal(dim)
            norm = np.linalg.norm(mutation_vectors[i])
            if norm > 0:
                mutation_vectors[i] /= norm

        # Initial sigma for DES
        sigma_des = 1.0
        min_sigma_des = 0.001
        max_sigma_des = 3.0

        # Run DES for remaining budget
        while calls_made < budget - 5 and time.time() < deadline:
            # Generate offspring using deterministic mutation vectors
            offspring = np.zeros((pop_size, dim))
            for i in range(pop_size):
                offspring[i] = x_global_best + sigma_des * mutation_vectors[i]
                offspring[i] = np.clip(offspring[i], bounds, upper)
            
            # Evaluate offspring
            best_offspring_x = None
            best_offspring_f = np.inf
            for i in range(pop_size):
                if calls_made >= budget - 5:
                    break
                if time.time() > deadline:
                    return x_global_best, fx_global_best
                
                x = offspring[i]
                fx = func(x)
                calls_made += 1
                update_best(x, fx)
                
                if fx < best_offspring_f:
                    best_offspring_f = fx
                    best_offspring_x = x.copy()
            
            # Adapt sigma based on success rate
            # If we found a better point, increase sigma slightly
            # Otherwise, decrease sigma to refine
            if best_offspring_f < fx_global_best:
                sigma_des = min(sigma_des * 1.1, max_sigma_des)
            else:
                sigma_des = max(sigma_des * 0.8, min_sigma_des)
                
            # If sigma is very small, break to final polish
            if sigma_des <= min_sigma_des * 1.01:
                break

        # Phase 3: Final polish with (1+1)-ES
        remaining = budget - calls_made
        if remaining > 10 and x_global_best is not None:
            x = x_global_best.copy()
            fx = fx_global_best
            sigma_polish = 0.05
            min_sigma_polish = 0.0005
            
            while calls_made < budget - 2 and time.time() < deadline:
                y = x + sigma_polish * rng.standard_normal(dim)
                y = np.clip(y, bounds, upper)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma_polish = min(sigma_polish * 1.1, 0.2)
                else:
                    sigma_polish = max(sigma_polish * 0.7, min_sigma_polish)
                    
                    if sigma_polish <= min_sigma_polish * 1.01:
                        y2 = x + 0.01 * rng.standard_normal(dim)
                        y2 = np.clip(y2, bounds, upper)
                        fy2 = func(y2)
                        calls_made += 1
                        if fy2 < fx:
                            x, fx = y2, fy2
                            update_best(x, fx)
                            sigma_polish = 0.02
                        else:
                            break

        return x_global_best, fx_global_best
# EVOLVE-BLOCK-END
