"""Multi-objective optimizer for ZDT6 using NSGA-II style search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective search with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        pop_size = min(50, max(10, self.budget // 4))
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = [[]]
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                    elif dominates(F[q], F[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    fronts[0].append(p)
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, front):
            if len(front) <= 2:
                return {idx: float('inf') for idx in front}
            m = self.n_obj
            dist = {idx: 0.0 for idx in front}
            for m_idx in range(m):
                sorted_front = sorted(front, key=lambda idx: F[idx, m_idx])
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')
                f_min = F[sorted_front[0], m_idx]
                f_max = F[sorted_front[-1], m_idx]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                for k in range(1, len(sorted_front) - 1):
                    dist[sorted_front[k]] += (F[sorted_front[k + 1], m_idx] - F[sorted_front[k - 1], m_idx]) / f_range
            return dist

        while used < self.budget and time.time() < deadline:
            fronts = fast_non_dominated_sort(F)
            
            # Create offspring
            offspring = []
            offspring_F = []
            while len(offspring) < pop_size and time.time() < deadline:
                # Binary tournament
                p1_idx = rng.integers(pop_size)
                p2_idx = rng.integers(pop_size)
                p1_front = next(i for i, f in enumerate(fronts) if p1_idx in f)
                p2_front = next(i for i, f in enumerate(fronts) if p2_idx in f)
                if p1_front < p2_front:
                    parent = p1_idx
                elif p2_front < p1_front:
                    parent = p2_idx
                else:
                    # Same front, use crowding distance
                    cd = crowding_distance(F, fronts[p1_front])
                    if cd[p1_idx] >= cd[p2_idx]:
                        parent = p1_idx
                    else:
                        parent = p2_idx
                
                # Mutation
                sigma = 0.2
                child = X[parent].copy()
                mutation_mask = rng.random(self.dim) < 0.3
                child[mutation_mask] += sigma * span[mutation_mask] * rng.standard_normal(np.sum(mutation_mask))
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                offspring.append(child)
                offspring_F.append(fc)
            
            if not offspring:
                break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            X_comb = np.vstack([X, np.array(offspring)])
            F_comb = np.vstack([F, offspring_F])
            
            fronts_comb = fast_non_dominated_sort(F_comb)
            
            new_X = []
            new_F = []
            for front in fronts_comb:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_comb, front)
                    sorted_front = sorted(front, key=lambda idx: cd[idx], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return X
# EVOLVE-BLOCK-END
