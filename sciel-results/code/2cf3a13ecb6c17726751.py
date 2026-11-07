import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 325.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []

            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                        n_dom[q] += 1
                    elif dominates(F[q], F[p]):
                        S[q].append(p)
                        n_dom[p] += 1

            fronts = []
            current_front = []
            for p in range(n):
                if n_dom[p] == 0:
                    rank[p] = 0
                    current_front.append(p)

            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = rank[p] + 1
                            next_front.append(q)
                current_front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            if not front_indices:
                return {}

            m = len(front_indices)
            if m <= 2:
                return {i: float('inf') for i in front_indices}

            dist = {i: 0.0 for i in front_indices}

            for k in range(n_obj):
                sorted_front = sorted(front_indices, key=lambda i: F[i][k])

                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')

                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                f_range = f_max - f_min

                if f_range == 0:
                    continue

                for i in range(1, m - 1):
                    dist[sorted_front[i]] += (F[sorted_front[i+1]][k] - F[sorted_front[i-1]][k]) / f_range

            return dist

        def compare(i, j, rank_map, crowd_dists):
            rank_i = rank_map.get(i, float('inf'))
            rank_j = rank_map.get(j, float('inf'))

            if rank_i < rank_j:
                return True
            elif rank_i > rank_j:
                return False
            else:
                dist_i = crowd_dists.get(i, 0.0)
                dist_j = crowd_dists.get(j, 0.0)
                return dist_i > dist_j

        # --- Main Algorithm: Differential Evolution with Adaptive Mutation and Pareto Selection ---
        
        # Population size
        N = 20
        CR = 0.8
        F_factor = 0.5

        # 1. Initialize population
        pop_X = []
        pop_F = []

        for _ in range(N):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            f = evaluate(x)
            pop_X.append(x)
            pop_F.append(f)

        # 2. Evolution loop
        while used < budget and time.time() < deadline:
            offspring_X = []
            offspring_F = []

            # Pre-compute ranks and crowding distances for current population to speed up selection
            # Note: In standard DE, we don't necessarily need complex selection for parent generation,
            # but we do need it for the final selection. However, using ranks for parent selection
            # (elitism) can help. Let's stick to standard DE parent selection (random) but use
            # ranks for the final (mu+lambda) selection.
            
            # Actually, the inspiration uses standard DE parent selection (random distinct indices).
            # Let's use that.
            
            for i in range(N):
                if used >= budget or time.time() > deadline:
                    break
                
                # Pick 3 distinct random indices from 0 to N-1, excluding i
                candidates = [j for j in range(N) if j != i]
                if len(candidates) < 3:
                    # Fallback if N is very small
                    r1, r2, r3 = np.random.choice(N, size=3, replace=True)
                    if r1 == i: r1 = (r1 + 1) % N
                    if r2 == i: r2 = (r2 + 1) % N
                    if r3 == i: r3 = (r3 + 1) % N
                else:
                    r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                
                # Differential Evolution: v = x_r1 + F * (x_r2 - x_r3)
                v = pop_X[r1] + F_factor * (pop_X[r2] - pop_X[r3])
                
                # Binomial Crossover
                trial = np.zeros(dim)
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop_X[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                f = evaluate(trial)
                offspring_X.append(trial)
                offspring_F.append(f)
            
            if not offspring_X:
                break
                
            # Combine parents and offspring
            combined_X = pop_X + offspring_X
            combined_F = pop_F + offspring_F
            
            # Sort combined population
            fronts = non_dominated_sort(np.array(combined_F))
            rank_map = {}
            for rank, front in enumerate(fronts):
                for idx in front:
                    rank_map[idx] = rank
            
            crowd_dists = {}
            for front in fronts:
                dists = crowding_distance(np.array(combined_F), front)
                crowd_dists.update(dists)
            
            # Select next population of size N
            new_pop_X = []
            new_pop_F = []
            
            for rank, front in enumerate(fronts):
                if len(new_pop_X) + len(front) <= N:
                    for idx in front:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                else:
                    remaining = N - len(new_pop_X)
                    sorted_front = sorted(front, key=lambda i: crowd_dists.get(i, 0.0), reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_pop_X.append(combined_X[idx])
                        new_pop_F.append(combined_F[idx])
                    break
            
            pop_X = new_pop_X
            pop_F = new_pop_F

        return all_X
