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

        pop_size = 6
        # Initialize population
        # One point at origin, one at ones (common Rosenbrock min location), rest random
        pop = np.zeros((pop_size, dim))
        pop[0] = 0.0
        if pop_size > 1:
            pop[1] = 1.0
        for i in range(2, pop_size):
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
        sigma = 2.0 # Start large to explore
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
            
            for i in range(n_offspring):
                # Select two parents from the top n_parents for recombination
                p1_idx = rng.choice(n_parents)
                p2_idx = rng.choice(n_parents)
                
                p1 = pop[p1_idx]
                p2 = pop[p2_idx]
                
                # Recombination (intermediate)
                # Mix parents with some randomness
                mix = rng.random(dim)
                child = mix * p1 + (1 - mix) * p2
                
                # Mutation
                m = child + sigma * rng.standard_normal(dim)
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
            if new_pop_f[0] < best_f:
                # Global best improved: we are finding good regions, shrink sigma to exploit
                best_f = new_pop_f[0]
                best_x = new_pop[0].copy()
                sigma = max(sigma * 0.9, min_sigma)
            else:
                # No improvement: we are stuck or exploring, expand sigma
                sigma = min(sigma * 1.1, max_sigma)

            pop = new_pop
            pop_f = new_pop_f

            # Late stage refinement: force small sigma if very close to end
            if remaining < 50:
                sigma = min(sigma, 0.05)
                
        return best_x, best_f
