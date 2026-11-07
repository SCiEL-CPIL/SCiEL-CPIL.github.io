"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Initial population
        pop_size = 50
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = np.zeros(n, dtype=int)
            front = [0]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dominates[p] += 1
                if n_dominates[p] == 0:
                    front[0] += 1
                    if front[0] == 1:
                        front = [0] * (n + 1)
                    front[0] = 0  # reset for count
            
            # Correct implementation
            ranks = np.zeros(n, dtype=int)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            front = [0]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[p] += 1
                if n_dom[p] == 0:
                    front.append(p)
                    ranks[p] = 0
            
            i = 0
            while len(front) > 0:
                i += 1
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            ranks[q] = i
                            next_front.append(q)
                front = next_front
            
            return ranks

        def crowding_distance(F, ranks):
            n = len(F)
            cd = np.zeros(n)
            unique_ranks = np.unique(ranks)
            
            for rank in unique_ranks:
                members = np.where(ranks == rank)[0]
                if len(members) <= 2:
                    cd[members] = np.inf
                    continue
                
                for m in range(self.n_obj):
                    obj_vals = F[members, m]
                    sorted_idx = np.argsort(obj_vals)
                    cd[members[sorted_idx[0]]] = np.inf
                    cd[members[sorted_idx[-1]]] = np.inf
                    obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                    if obj_range > 1e-12:
                        for k in range(1, len(sorted_idx) - 1):
                            cd[members[sorted_idx[k]]] += (
                                obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]
                            ) / obj_range
            
            return cd

        # Main evolution loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            ranks = fast_non_dominated_sort(F)
            cd = crowding_distance(F, ranks)
            
            # Select parents using tournament selection
            parent1_idx = rng.integers(pop_size)
            parent2_idx = rng.integers(pop_size)
            
            # Tournament
            for _ in range(3):
                cand = rng.integers(pop_size)
                if ranks[cand] < ranks[parent1_idx] or (ranks[cand] == ranks[parent1_idx] and cd[cand] > cd[parent1_idx]):
                    parent1_idx = cand
            
            for _ in range(3):
                cand = rng.integers(pop_size)
                if ranks[cand] < ranks[parent2_idx] or (ranks[cand] == ranks[parent2_idx] and cd[cand] > cd[parent2_idx]):
                    parent2_idx = cand
            
            # Crossover (SBX)
            if rng.random() < 0.9:
                x1 = X[parent1_idx]
                x2 = X[parent2_idx]
                beta = 15
                child1 = np.zeros(self.dim)
                child2 = np.zeros(self.dim)
                for i in range(self.dim):
                    u = rng.random()
                    if u <= 0.5:
                        alpha = 1 + 2 * (x1[i] - x2[i]) / (hi[i] - lo[i]) if hi[i] != lo[i] else 1
                        beta_c = (1 + 1 / beta) ** (-1) if u <= 0.5 else (2 * u) ** (-1)
                        child1[i] = 0.5 * ((1 + beta_c) * x1[i] + (1 - beta_c) * x2[i])
                    else:
                        alpha = 1 + 2 * (x2[i] - x1[i]) / (hi[i] - lo[i]) if hi[i] != lo[i] else 1
                        beta_c = (1 + 1 / beta) ** (-1) if u > 0.5 else (2 * (1 - u)) ** (-1)
                        child1[i] = 0.5 * ((1 + beta_c) * x2[i] + (1 - beta_c) * x1[i])
                    child2[i] = hi[i] + lo[i] - child1[i]
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
            else:
                child1 = x1.copy()
                child2 = x2.copy()
            
            # Mutation
            for child in [child1, child2]:
                for i in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        sigma = 0.2 * (hi[i] - lo[i])
                        child[i] = child[i] + sigma * rng.standard_normal()
                        child[i] = np.clip(child[i], lo[i], hi[i])
            
            # Evaluate children
            f1 = problem(child1)
            used += 1
            f2 = problem(child2)
            used += 1
            
            # Combine parent and child populations
            X_new = np.vstack([X, child1, child2])
            F_new = np.vstack([F, f1, f2])
            
            # Selection: keep best pop_size
            ranks_new = fast_non_dominated_sort(F_new)
            cd_new = crowding_distance(F_new, ranks_new)
            
            # Sort by rank then crowding distance
            sort_idx = np.lexsort((cd_new, ranks_new))
            X = X_new[sort_idx[:pop_size]]
            F = F_new[sort_idx[:pop_size]]
            
            # Adjust used count (we added 2 but kept same pop size)
            # Actually used already incremented, so we're fine
            
            # Early exit if time is tight
            if time.time() > deadline:
                break
        
        return X
# EVOLVE-BLOCK-END
