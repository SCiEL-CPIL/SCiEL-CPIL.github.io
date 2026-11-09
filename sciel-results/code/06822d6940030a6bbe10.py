"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        n = self.budget
        pop_size = min(60, max(20, n // 3))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Non-dominated sorting (fast NSGA-II style)
        def non_dominated_sort(F):
            npts = F.shape[0]
            S = [[] for _ in range(npts)]
            n_dom = np.zeros(npts, dtype=int)
            fronts = [[]]
            for i in range(npts):
                for j in range(npts):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)
            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if len(next_front) > 0:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, front_indices):
            npts = len(front_indices)
            if npts <= 2:
                return np.full(npts, np.inf)
            cd = np.zeros(npts)
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F[front_indices, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                fmin = F[front_indices[sorted_idx[0]], m]
                fmax = F[front_indices[sorted_idx[-1]], m]
                if fmax - fmin < 1e-10:
                    continue
                for k in range(1, npts - 1):
                    cd[sorted_idx[k]] += (F[front_indices[sorted_idx[k+1]], m] - F[front_indices[sorted_idx[k-1]], m]) / (fmax - fmin)
            return cd

        # Main loop
        while used < n:
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            fronts = non_dominated_sort(F)
            
            # Build mating pool from first fronts
            mating_pool = []
            for front in fronts:
                mating_pool.extend(front)
                if len(mating_pool) >= pop_size:
                    break
            
            # Generate offspring using DE/rand/1
            offspring_X = []
            offspring_F = []
            for _ in range(pop_size):
                if len(mating_pool) < 3:
                    # Fallback to random mutation
                    p1_idx = rng.integers(len(mating_pool))
                    parent = X[mating_pool[p1_idx]]
                    child = parent + 0.2 * (hi - lo) * rng.standard_normal(self.dim)
                else:
                    # DE/rand/1
                    idxs = rng.choice(len(mating_pool), size=3, replace=False)
                    a, b, c = mating_pool[idxs[0]], mating_pool[idxs[1]], mating_pool[idxs[2]]
                    CR = 0.9
                    F_de = 0.8
                    v = X[a] + F_de * (X[b] - X[c])
                    # Crossover
                    j_rand = rng.integers(self.dim)
                    mask = rng.random(self.dim) < CR
                    mask[j_rand] = True
                    child = np.where(mask, v, X[a])
                
                child = np.clip(child, lo, hi)
                offspring_X.append(child)
                offspring_F.append(problem(child))
                used += 1
                if used >= n or time.time() > deadline:
                    break
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring populations
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_size = combined_X.shape[0]
            
            # Non-dominated sort on combined population
            combined_fronts = non_dominated_sort(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to truncate this front using crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[sorted_by_cd[i]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
        
        return X
# EVOLVE-BLOCK-END
