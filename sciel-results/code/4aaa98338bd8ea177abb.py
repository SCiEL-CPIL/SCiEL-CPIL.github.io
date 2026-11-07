import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Differential Evolution (DE) with 
    Non-Dominated Sorting selection (MOEA/DE)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Population size
        # 200 budget. Let's use N=40 to get 4 generations.
        # 40 init + 40*4 = 200.
        pop_size = 40
        if self.budget < pop_size:
            pop_size = max(2, self.budget)

        # Initialize population using random sampling
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominating = np.zeros(n, dtype=int)
            fronts = []
            front0 = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif dominates(F[j], F[i]):
                        n_dominating[i] += 1
                if n_dominating[i] == 0:
                    front0.append(i)

            fronts.append(front0)
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front0 = next_front
            return fronts

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[front_indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-15:
                    continue
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_v - min_v)
            return cd

        def select_next_population(F_all, X_all, pop_size):
            """Select next population using NSGA-II logic."""
            fronts = fast_non_dominated_sort(F_all)
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    remaining = pop_size - len(new_indices)
                    if remaining <= 0:
                        break
                    cd = crowding_distance(F_all, front)
                    # Sort by crowding distance descending
                    sorted_front = [front[j] for j in np.argsort(-cd)]
                    new_indices.extend(sorted_front[:remaining])
                    break
            
            # Safety check
            while len(new_indices) < pop_size:
                new_indices.append(rng.integers(len(F_all)))
            
            return X_all[new_indices], F_all[new_indices]

        # DE Parameters
        F_scale = 0.8  # Differential weight
        CR = 0.5       # Crossover rate

        while used < self.budget:
            if time.time() > deadline:
                break

            n_children = min(pop_size, self.budget - used)
            if n_children <= 0:
                break

            X_c = np.zeros((n_children, self.dim))
            
            # Generate children using DE
            for i in range(n_children):
                # Select 3 distinct random individuals for mutation
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = X[r1] + F_scale * (X[r2] - X[r3])
                
                # Clip to bounds immediately to avoid invalid values
                v = np.clip(v, lo, hi)
                
                # Binomial Crossover
                # Pick a random index for the trial vector to be different
                j_rand = rng.integers(self.dim)
                trial = X[i].copy() # Start with parent x_i
                
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                X_c[i] = trial

            F_c = np.array([problem(x) for x in X_c])
            used += n_children

            # Combine and Select
            X_all = np.vstack([X, X_c])
            F_all = np.vstack([F, F_c])
            
            X, F = select_next_population(F_all, X_all, pop_size)

        return X
