# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic hierarchical search: coarse grid scan then local CMA-ES for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse grid scan to identify global basin
        # Use 3 points per dimension: -4, 0, 4 -> 3^5 = 243 points
        # Shuffle to ensure early progress
        grid_vals = [-4.0, 0.0, 4.0]
        grid = np.array(np.meshgrid(*([grid_vals] * dim))).T.reshape(-1, dim)
        grid_order = rng.permutation(len(grid))
        
        n_grid = min(243, budget)
        for idx in grid_order[:n_grid]:
            if calls >= budget or time.time() > deadline:
                break
            x = grid[idx]
            fx = evaluate(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return None, np.inf

        # Phase 2: Local CMA-ES refinement around best point
        # Allocate remaining budget to local search
        remaining = budget - calls
        sigma = 1.0  # Initial step size for local search
        
        while calls < budget and time.time() < deadline:
            # Generate population around best_x
            pop_size = min(20, remaining)
            if pop_size < 2:
                break
                
            offspring = []
            fitnesses = []
            
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                y = rng.standard_normal(dim)
                xi = best_x + sigma * y
                xi = np.clip(xi, -5.0, 5.0)
                fx = evaluate(xi)
                if fx is None:
                    break
                offspring.append(xi)
                fitnesses.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = xi.copy()
            
            if len(offspring) < 2:
                break
            
            # Sort by fitness
            indices = np.argsort(fitnesses)
            
            # Update center to weighted mean of best half
            mu_half = max(2, len(offspring) // 2)
            best_x_new = np.mean([offspring[indices[i]] for i in range(mu_half)], axis=0)
            
            # Adaptive step size: shrink if improving, expand if not
            best_fitness_now = fitnesses[indices[0]]
            if best_fitness_now < best_f + 1e-10:
                best_x = best_x_new
                sigma *= 0.7  # Shrink step size on improvement
            else:
                sigma *= 1.3  # Expand step size on stagnation
                sigma = min(sigma, 2.0)  # Cap step size
            
            # Check for convergence
            if sigma < 0.01:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
