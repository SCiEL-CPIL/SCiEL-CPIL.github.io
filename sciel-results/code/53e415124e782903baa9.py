"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _fast_non_dominated(self, F):
        """Return list of fronts (each a list of indices) using non-dominated sorting."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = [0] * n
        fronts = []
        front_idx = [0] * n
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                front_idx[i] = 0
                fronts.append([]) if not fronts else None
                if not fronts:
                    fronts.append([])
                fronts[0].append(i)
        
        k = 0
        while fronts[k] and k < n:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        front_idx[j] = k + 1
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return [float('inf')] * n
        
        dist = [0.0] * n
        m = self.n_obj
        
        for k in range(m):
            sorted_idx = sorted(indices, key=lambda i: F[i][k])
            dist[sorted_idx[0]] = float('inf')
            dist[sorted_idx[-1]] = float('inf')
            f_min = F[sorted_idx[0]][k]
            f_max = F[sorted_idx[-1]][k]
            if f_max == f_min:
                continue
            for i in range(1, n - 1):
                idx = sorted_idx[i]
                dist[idx] += (F[sorted_idx[i + 1]][k] - F[sorted_idx[i - 1]][k]) / (f_max - f_min)
        
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = min(40, max(10, self.budget // 5))
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated(F)
            
            # Create offspring
            offspring = []
            offspring_F = []
            n_offspring = pop_size
            
            while len(offspring) < n_offspring:
                if time.time() > deadline:
                    break
                
                # Select parents from fronts
                parents = []
                for front in fronts:
                    if len(parents) + len(front) >= 2:
                        # Crowding distance for selection
                        cd = self._crowding_distance(F, front)
                        # Pick 2 parents with highest crowding distance
                        front_indices = np.array(front)
                        cd_arr = np.array(cd)
                        # Sort by crowding distance descending
                        order = np.argsort(-cd_arr)
                        for idx in order:
                            if len(parents) < 2:
                                parents.append(front_indices[idx])
                            else:
                                break
                        break
                    else:
                        parents.extend(front)
                        if len(parents) >= 2:
                            break
                
                if len(parents) < 2:
                    parents = [0, 1]
                
                p1, p2 = parents[0], parents[1]
                
                # SBX crossover
                child1 = X[p1].copy()
                child2 = X[p2].copy()
                if rng.random() < 0.9:
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / 20)
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / 20)
                            child1[d] = 0.5 * ((1 + beta) * X[p1][d] + (1 - beta) * X[p2][d])
                            child2[d] = 0.5 * ((1 - beta) * X[p1][d] + (1 + beta) * X[p2][d])
                
                # Mutation
                for child in [child1, child2]:
                    if rng.random() < 0.9:
                        sigma = 0.1 * (hi - lo)
                        child = child + sigma * rng.standard_normal(self.dim)
                        child = np.clip(child, lo, hi)
                
                offspring.append(child1)
                offspring_F.append(problem(child1))
                used += 1
                if used >= self.budget or time.time() > deadline:
                    break
                
                offspring.append(child2)
                offspring_F.append(problem(child2))
                used += 1
            
            if not offspring:
                break
            
            # Combine parent and offspring
            X_comb = np.vstack([X, np.array(offspring)])
            F_comb = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sort combined
            fronts_comb = self._fast_non_dominated(F_comb)
            
            # Select next generation
            X_new = []
            F_new = []
            for front in fronts_comb:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    remaining = pop_size - len(X_new)
                    cd = self._crowding_distance(F_comb, front)
                    front_indices = np.array(front)
                    cd_arr = np.array(cd)
                    order = np.argsort(-cd_arr)
                    for idx in order[:remaining]:
                        X_new.append(X_comb[front_indices[idx]])
                        F_new.append(F_comb[front_indices[idx]])
                    break
            
            if X_new:
                X = np.array(X_new)
                F = np.array(F_new)
        
        return X
# EVOLVE-BLOCK-END
