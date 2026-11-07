import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290

        pop_size = 24
        if pop_size > budget:
            pop_size = budget

        # Initialize
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # DE Parameters
        F_de = 0.8
        CR = 0.9

        while used + pop_size <= budget:
            if time.time() > deadline:
                break

            # Generate Offspring using DE/rand/1/bin
            offspring_X = np.empty((pop_size, dim))
            for i in range(pop_size):
                # Choose 3 distinct random parents
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = X[r1] + F_de * (X[r2] - X[r3])
                
                # Crossover
                c = X[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        c[j] = v[j]
                
                # Clip to bounds
                c = np.clip(c, lo, hi)
                offspring_X[i] = c

            # Evaluate Offspring
            F_off = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += pop_size

            # Combine
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])

            # Non-dominated Sort
            ranks, fronts = self.non_dominated_sort(F_all)
            
            # Crowding Distance
            cd = self.crowding_distance(F_all, fronts)

            # Selection
            sort_keys = np.lexsort((-cd, ranks))
            next_idx = sort_keys[:pop_size]
            
            X = X_all[next_idx]
            F = F_all[next_idx]

        return X

    # ... (keep non_dominated_sort, crowding_distance, dominates from original)
