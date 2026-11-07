# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Differential Evolution (MODE) optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        # With budget 200, we can afford a larger population for better initial coverage.
        # N=100 allows 2 generations (100 init + 100 offspring).
        # N=50 allows 4 generations.
        # DE often benefits from larger populations. Let's try N=100.
        N = 100
        if budget < 2 * N:
            N = max(10, budget // 2)

        # Initial population
        X = lo + span * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            for i in current_front:
                rank[i] = 0
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, indices):
            if not indices:
                return []
            
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[indices, m]
                f_min = np.min(f_vals)
                f_max = np.max(f_vals)
                if f_max - f_min < 1e-12:
                    continue
                sorted_idx = np.argsort(f_vals)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (f_vals[sorted_idx[i + 1]] - f_vals[sorted_idx[i - 1]]) / (f_max - f_min)
            
            return dist

        def select_parents(pop_indices, ranks, crowd_dists):
            def tournament():
                a = rng.integers(len(pop_indices))
                b = rng.integers(len(pop_indices))
                ia = pop_indices[a]
                ib = pop_indices[b]
                if ranks[ia] < ranks[ib]:
                    return ia
                elif ranks[ib] < ranks[ia]:
                    return ib
                else:
                    if crowd_dists[ia] > crowd_dists[ib]:
                        return ia
                    else:
                        return ib
            
            p1 = tournament()
            p2 = tournament()
            return p1, p2

        # DE Parameters
        F_mut = 0.8  # Mutation factor
        CR = 0.9     # Crossover rate

        while used < budget:
            # Calculate ranks and crowding distances for current population
            fronts, ranks = fast_non_dominated_sort(F)
            
            crowd_dists = np.zeros(N)
            for front in fronts:
                cd = crowding_distance(F, front)
                for idx, dist in zip(front, cd):
                    crowd_dists[idx] = dist

            # Generate offspring using DE
            n_offspring = min(N, budget - used)
            offspring_X = np.zeros((n_offspring, dim))
            offspring_F = np.zeros((n_offspring, n_obj))
            
            for i in range(n_offspring):
                # Select 3 distinct parents
                r1, r2, r3 = rng.choice(N, 3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = X[r1] + F_mut * (X[r2] - X[r3])
                
                # Crossover: Binomial
                crossover_point = rng.integers(dim)
                trial = np.copy(X[i]) # Target is current individual i? Or random?
                # Standard DE uses the current individual as target.
                # But we are generating a new population. 
                # Let's use the current individual X[i] as the target for variation.
                
                for j in range(dim):
                    if rng.random() < CR or j == crossover_point:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                offspring_X[i] = trial
                offspring_F[i] = problem(trial)
                used += 1
                
                if used >= budget:
                    break

            # Merge populations
            X_new = np.vstack([X, offspring_X])
            F_new = np.vstack([F, offspring_F])
            
            # Select top N from merged population using NSGA-II selection
            fronts_new, ranks_new = fast_non_dominated_sort(F_new)
            
            selected_indices = []
            for front in fronts_new:
                if len(selected_indices) + len(front) <= N:
                    selected_indices.extend(front)
                else:
                    remaining = N - len(selected_indices)
                    cd = crowding_distance(F_new, front)
                    sorted_front = sorted(zip(front, cd), key=lambda x: x[1], reverse=True)
                    selected_indices.extend([idx for idx, _ in sorted_front[:remaining]])
                    break
            
            # Update population
            X = X_new[selected_indices]
            F = F_new[selected_indices]

        return X
# EVOLVE-BLOCK-END
