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

        # Parameters
        n_init = 20
        n_min = 5
        
        # Initialize population
        pop = np.zeros((n_init, dim))
        pop[0] = 0.0
        for i in range(1, n_init):
            pop[i] = rng.uniform(-5, 5, dim)
            
        pop_f = np.array([func(x) for x in pop])
        calls_made = n_init
        remaining = budget - calls_made

        # Sort
        idx = np.argsort(pop_f)
        pop = pop[idx]
        pop_f = pop_f[idx]
        
        best_x = pop[0].copy()
        best_f = pop_f[0]

        # Adaptive Sigma
        sigma = 2.0
        min_sigma = 1e-6
        max_sigma = 10.0
        target_success_rate = 0.2
        
        while remaining > 0:
            # Determine current population size (linear decay)
            progress = 1.0 - (remaining / (budget - n_init)) if budget > n_init else 0.0
            progress = np.clip(progress, 0.0, 1.0)
            n_cur = int(np.round(n_init + (n_min - n_init) * progress))
            n_cur = max(n_cur, n_min)
            
            # Shrink population if needed
            if len(pop) > n_cur:
                pop = pop[:n_cur]
                pop_f = pop_f[:n_cur]

            n_offspring = n_cur
            if n_offspring > remaining:
                n_offspring = remaining
                if n_offspring == 0:
                    break
            remaining -= n_offspring

            # Selection: Rank-based weighted selection
            weights = np.linspace(len(pop), 1, len(pop))
            weights /= np.sum(weights)
            parent_idx = rng.choice(len(pop), size=n_offspring, replace=True, p=weights)
            
            offspring = np.zeros((n_offspring, dim))
            offspring_f = np.zeros(n_offspring)
            successes = 0
            
            for i in range(n_offspring):
                p = pop[parent_idx[i]]
                m = p + sigma * rng.standard_normal(dim)
                m = np.clip(m, -5, 5)
                offspring[i] = m
                f_val = func(m)
                offspring_f[i] = f_val
                
                if f_val < pop_f[parent_idx[i]]:
                    successes += 1

            # Combine and select
            all_x = np.vstack([pop, offspring])
            all_f = np.concatenate([pop_f, offspring_f])
            
            sort_idx = np.argsort(all_f)
            new_pop = all_x[sort_idx[:n_cur]].copy()
            new_pop_f = all_f[sort_idx[:n_cur]].copy()
            
            # Update global best
            if new_pop_f[0] < best_f:
                best_f = new_pop_f[0]
                best_x = new_pop[0].copy()

            # Adapt Sigma
            if n_offspring > 0:
                success_rate = successes / n_offspring
                if success_rate > target_success_rate * 1.5:
                    sigma *= 1.1
                elif success_rate < target_success_rate * 0.5:
                    sigma *= 0.9
                sigma = np.clip(sigma, min_sigma, max_sigma)

            # Diversity Injection (Early phase only)
            if progress < 0.5 and remaining > 0 and n_cur > 1:
                pop_std = np.std(new_pop, axis=0)
                avg_std = np.mean(pop_std)
                if avg_std < 0.2:
                    worst_idx = np.argmax(new_pop_f)
                    new_pop[worst_idx] = rng.uniform(-5, 5, dim)
                    new_pop_f[worst_idx] = func(new_pop[worst_idx])
                    remaining -= 1
                    sort_idx = np.argsort(new_pop_f)
                    new_pop = new_pop[sort_idx[:n_cur]]
                    new_pop_f = new_pop_f[sort_idx[:n_cur]]
                    if new_pop_f[0] < best_f:
                        best_f = new_pop_f[0]
                        best_x = new_pop[0].copy()

            pop = new_pop
            pop_f = new_pop_f

        return best_x, best_f
