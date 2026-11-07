"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II based optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget
        
        # Population size
        pop_size = 20
        
        # Initialize population
        # For ZDT4, x1 is the shape variable, others should be near 0.
        # We use a mix of uniform and Gaussian for the tail.
        pop = np.zeros((pop_size, n))
        for i in range(pop_size):
            pop[i, 0] = np.random.uniform(lo[0], hi[0])
            for j in range(1, n):
                # Gaussian centered at 0, clipped to bounds
                val = np.random.normal(0, (hi[j] - lo[j]) / 4)
                pop[i, j] = np.clip(val, lo[j], hi[j])
        
        evals = 0
        
        # Evaluate initial population
        fits = np.array([problem(x) for x in pop])
        evals += pop_size
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(fits):
            m = len(fits)
            ranks = np.zeros(m, dtype=int)
            S = [set() for _ in range(m)] 
            n_d = np.zeros(m, dtype=int)
            
            for i in range(m):
                for j in range(i+1, m):
                    if np.all(fits[i] <= fits[j]) and np.any(fits[i] < fits[j]):
                        S[i].add(j)
                        n_d[j] += 1
                    elif np.all(fits[j] <= fits[i]) and np.any(fits[j] < fits[i]):
                        S[j].add(i)
                        n_d[i] += 1
            
            current_rank = 0
            front = [i for i in range(m) if n_d[i] == 0]
            while front:
                next_front = []
                for i in front:
                    ranks[i] = current_rank
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                front = next_front
                current_rank += 1
            return ranks

        # Helper: Crowding distance
        def crowding_distance(fits, ranks):
            m = len(fits)
            cd = np.zeros(m)
            unique_ranks = np.unique(ranks)
            for r in unique_ranks:
                idx = np.where(ranks == r)[0]
                if len(idx) <= 2:
                    cd[idx] = np.inf
                    continue
                for k in range(self.n_obj):
                    sorted_idx = idx[np.argsort(fits[idx, k])]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    f_range = fits[sorted_idx[-1], k] - fits[sorted_idx[0], k]
                    if f_range == 0:
                        continue
                    for i in range(1, len(sorted_idx)-1):
                        prev_idx = sorted_idx[i-1]
                        next_idx = sorted_idx[i+1]
                        cd[sorted_idx[i]] += (fits[next_idx, k] - fits[prev_idx, k]) / f_range
            return cd

        # Selection: Tournament
        def tournament_select(pop, fits, ranks, cd, k=2):
            indices = np.random.choice(len(pop), k, replace=False)
            best_idx = indices[0]
            for idx in indices[1:]:
                if ranks[idx] < ranks[best_idx]:
                    best_idx = idx
                elif ranks[idx] == ranks[best_idx] and cd[idx] > cd[best_idx]:
                    best_idx = idx
            return pop[best_idx].copy()

        # Crossover: Blend with noise
        def crossover(x1, x2, lo, hi):
            x = 0.5 * (x1 + x2)
            noise = np.random.normal(0, 0.1, size=n)
            x = x + noise
            x = np.clip(x, lo, hi)
            return x

        # Main loop
        while evals < budget:
            p1 = tournament_select(pop, fits, ranks, cd)
            p2 = tournament_select(pop, fits, ranks, cd)
            
            child = crossover(p1, p2, lo, hi)
            
            f_child = problem(child)
            evals += 1
            
            new_pop = np.vstack([pop, child.reshape(1, -1)])
            new_fits = np.vstack([fits, f_child.reshape(1, -1)])
            
            new_ranks = non_dominated_sort(new_fits)
            new_cd = crowding_distance(new_fits, new_ranks)
            
            sort_keys = [(new_ranks[i], -new_cd[i], i) for i in range(len(new_pop))]
            sort_keys.sort()
            
            selected_indices = [s[2] for s in sort_keys[:pop_size]]
            
            pop = new_pop[selected_indices]
            fits = new_fits[selected_indices]
            
            ranks = non_dominated_sort(fits)
            cd = crowding_distance(fits, ranks)

        return None

# EVOLVE-BLOCK-END
