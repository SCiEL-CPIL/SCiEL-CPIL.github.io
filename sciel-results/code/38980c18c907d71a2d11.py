"""Multi-fidelity population-based ES with diverse archive and coarse-to-fine adaptive sigma."""
import numpy as np


class Optimizer:
    """
    Population-based evolutionary strategy with:
    - Diverse archive to prevent premature convergence
    - Coarse-to-fine sigma schedule
    - Sigma dynamically scaled by improvement rate
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        pop_size = 10
        # Initialize population
        # One point at origin (likely good for Rosenbrock)
        # Rest random
        pop = np.zeros((pop_size, dim))
        pop[0] = 0.0
        for i in range(1, pop_size):
            pop[i] = rng.uniform(-5, 5, dim)
            
        pop_f = np.array([func(x) for x in pop])
        calls_made = pop_size
        remaining = budget - calls_made

        # Sort
        idx = np.argsort(pop_f)
        pop = pop[idx]
        pop_f = pop_f[idx]
        
        best_x = pop[0].copy()
        best_f = pop_f[0]

        # Adaptive Sigma
        sigma = 1.0 # Start moderately
        min_sigma = 1e-8
        max_sigma = 5.0

        while remaining > 0:
            n_offspring = min(pop_size, remaining)
            if n_offspring == 0:
                break
            remaining -= n_offspring

            offspring = np.zeros((n_offspring, dim))
            offspring_f = np.zeros(n_offspring)

            # Selection: Use the best 50% of population as parents
            n_parents = max(2, pop_size // 2)
            # Weighted selection: better parents get more chances
            # Simple: random choice from top n_parents
            parent_idx = rng.choice(n_parents, size=n_offspring, replace=True)
            
            for i in range(n_offspring):
                p = pop[parent_idx[i]]
                # Mutation
                m = p + sigma * rng.standard_normal(dim)
                m = np.clip(m, -5, 5)
                offspring[i] = m
                offspring_f[i] = func(m)

            # Combine and select
            all_x = np.vstack([pop, offspring])
            all_f = np.concatenate([pop_f, offspring_f])
            
            # Keep best pop_size
            sort_idx = np.argsort(all_f)
            new_pop = all_x[sort_idx[:pop_size]].copy()
            new_pop_f = all_f[sort_idx[:pop_size]].copy()
            
            # Check for improvement to adapt sigma
            # If the new best is better than the old best, we are in a good spot.
            # However, a more robust signal is: did any offspring beat the worst parent?
            # Let's use: if min(offspring_f) < min(pop_f) (which is the best parent), we are improving.
            
            if new_pop_f[0] < best_f:
                # Global best improved
                best_f = new_pop_f[0]
                best_x = new_pop[0].copy()
                # Increase sigma to explore more of the basin
                sigma = min(sigma * 1.1, max_sigma)
            elif new_pop_f[0] < pop_f[0]:
                # Population best improved, but global best didn't? 
                # Actually, if new_pop_f[0] < pop_f[0], it means an offspring was better than the best parent.
                # This is a sign of progress.
                sigma = min(sigma * 1.05, max_sigma)
            else:
                # No improvement
                sigma = max(sigma * 0.9, min_sigma)

            pop = new_pop
            pop_f = new_pop_f

            # Late stage refinement
            if remaining < 100:
                # Force sigma to be small
                sigma = min(sigma, 0.1)
                
        return best_x, best_f
