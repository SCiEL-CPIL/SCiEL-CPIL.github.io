"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        pop_size = 30
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def non_dominated_sort(F):
            """Fast non-dominated sorting (Deb et al. 2002)."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_d = np.zeros(n, dtype=int)
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_d[p] += 1
                if n_d[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while len(fronts[i]) > 0:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            
            return fronts
        
        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)
            
            distances = np.zeros(len(front_indices))
            F_front = F[front_indices]
            
            for m in range(self.n_obj):
                order = np.argsort(F_front[:, m])
                distances[order[0]] = np.inf
                distances[order[-1]] = np.inf
                min_val = F_front[order[0], m]
                max_val = F_front[order[-1], m]
                if max_val - min_val < 1e-12:
                    continue
                for k in range(1, len(front_indices) - 1):
                    distances[order[k]] += (F_front[order[k + 1], m] - F_front[order[k - 1], m]) / (max_val - min_val)
            
            return distances
        
        def tournament_select(fronts, k=2):
            """Tournament selection based on rank and crowding distance."""
            n = len(F)
            idx = rng.integers(n, size=k)
            best = idx[0]
            for i in idx[1:]:
                rank_i = next(j for j, f in enumerate(fronts) if i in f)
                rank_b = next(j for j, f in enumerate(fronts) if best in f)
                if rank_i < rank_b:
                    best = i
                elif rank_i == rank_b:
                    front = fronts[rank_i]
                    cd = crowding_distance(F, front)
                    cd_map = {front[j]: cd[j] for j in range(len(front))}
                    if cd_map[i] > cd_map[best]:
                        best = i
            return best
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            fronts = non_dominated_sort(F)
            
            # Select parents using tournament selection
            p1 = tournament_select(fronts)
            p2 = tournament_select(fronts)
            
            # SBX crossover
            if rng.random() < 0.9:
                child1 = np.zeros(self.dim)
                child2 = np.zeros(self.dim)
                for j in range(self.dim):
                    x1, x2 = X[p1, j], X[p2, j]
                    if abs(x1 - x2) < 1e-14:
                        child1[j] = x1
                        child2[j] = x2
                        continue
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / 20)
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / 20)
                    child1[j] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    child2[j] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            else:
                child1 = X[p1].copy()
                child2 = X[p2].copy()
            
            # Polynomial mutation with adaptive sigma
            for child in [child1, child2]:
                for j in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        sigma = 0.15 * (hi[j] - lo[j])
                        child[j] += sigma * rng.standard_normal()
            
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            
            # Evaluate children
            fc1 = problem(child1)
            used += 1
            if used >= self.budget or time.time() > deadline:
                # Replace worst point
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst], F[worst] = child1, fc1
                break
            
            fc2 = problem(child2)
            used += 1
            
            # Combine parent and child populations
            X_comb = np.vstack([X, child1.reshape(1, -1), child2.reshape(1, -1)])
            F_comb = np.vstack([F, fc1.reshape(1, -1), fc2.reshape(1, -1)])
            
            # Truncate to population size using non-dominated sorting + crowding
            fronts = non_dominated_sort(F_comb)
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    cd = crowding_distance(F_comb, front)
                    order = np.argsort(-cd)
                    remaining = pop_size - len(new_indices)
                    new_indices.extend([front[i] for i in order[:remaining]])
                    break
            
            X = X_comb[new_indices]
            F = F_comb[new_indices]
        
        return X
# EVOLVE-BLOCK-END
