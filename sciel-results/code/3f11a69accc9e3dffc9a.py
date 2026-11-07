import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 275.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        
        pop_size = 30
        # Use LHS for initialization
        sampler = scipy.stats.qmc.LatinHypercube(d=dim, seed=12345)
        # Generate pop_size points in [0,1]
        points_01 = sampler.random(n=pop_size)
        X = lo + (hi - lo) * points_01
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper functions
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts.append([i])
            k = 0
            while k < len(fronts):
                current = fronts[k]
                next_front = []
                for i in current:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F_sub):
            m = len(F_sub)
            if m <= 2:
                return np.full(m, np.inf)
            dist = np.zeros(m)
            for k in range(n_obj):
                order = np.argsort(F_sub[:, k])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                fmin, fmax = F_sub[order[0], k], F_sub[order[-1], k]
                if fmax - fmin < 1e-14: continue
                for i in range(1, m - 1):
                    dist[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / (fmax - fmin)
            return dist

        def tournament_select(pop_X, pop_F, pop_fronts, pop_cd, k=2):
            candidates = np.random.randint(len(pop_X), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if pop_fronts[c] < pop_fronts[best]:
                    best = c
                elif pop_fronts[c] == pop_fronts[best] and pop_cd[c] > pop_cd[best]:
                    best = c
            return pop_X[best].copy()

        # Main Evolution Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Selection
            fronts = fast_non_dominated_sort(F)
            front_of = np.zeros(pop_size, dtype=int)
            for fi, front in enumerate(fronts):
                for idx in front:
                    front_of[idx] = fi
            cd = np.zeros(pop_size)
            for fi, front in enumerate(fronts):
                if len(front) > 0:
                    cd[front] = crowding_distance(F[front])
            
            p1 = tournament_select(X, F, front_of, cd)
            p2 = tournament_select(X, F, front_of, cd)
            p3 = tournament_select(X, F, front_of, cd)
            
            # DE Mutation
            F_scale = 0.7
            CR = 0.6
            v = p1 + F_scale * (p2 - p3)
            mask = np.random.random(dim) < CR
            mask[np.random.randint(dim)] = True
            child = np.where(mask, v, p1)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # NSGA-II Replacement
            # Combine
            X_new = np.vstack([X, [child]])
            F_new = np.vstack([F, [fc]])
            
            # Sort
            fronts_new = fast_non_dominated_sort(F_new)
            
            # Select next population
            X_next = []
            F_next = []
            for front in fronts_new:
                if len(X_next) + len(front) <= pop_size:
                    for idx in front:
                        X_next.append(X_new[idx])
                        F_next.append(F_new[idx])
                else:
                    # Need to truncate
                    remaining = pop_size - len(X_next)
                    F_front = F_new[front]
                    cd_front = crowding_distance(F_front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd_front)
                    for i in range(remaining):
                        idx = front[order[i]]
                        X_next.append(X_new[idx])
                        F_next.append(F_new[idx])
                    break
            
            X = np.array(X_next)
            F = np.array(F_next)
        
        # Polish Phase
        if used < self.budget and len(F) > 0:
            fronts = fast_non_dominated_sort(F)
            first_front_idx = fronts[0]
            first_front_F = F[first_front_idx]
            first_front_X = X[first_front_idx]
            
            # Pick diverse points from first front
            cd = crowding_distance(first_front_F)
            order = np.argsort(-cd)
            num_polish = min(5, len(order))
            polish_indices = order[:num_polish]
            
            local_sigma = 0.02
            for pi in polish_indices:
                if used >= self.budget: break
                if time.time() > deadline: break
                
                base_X = first_front_X[pi].copy()
                for step in range(6):
                    if used >= self.budget: break
                    if time.time() > deadline: break
                    
                    candidate = base_X + local_sigma * (hi - lo) * np.random.standard_normal(dim)
                    candidate = np.clip(candidate, lo, hi)
                    fc = problem(candidate)
                    used += 1
                    
                    # Check if non-dominated w.r.t current best front
                    dominated = False
                    for af in first_front_F:
                        if dominates(af, fc):
                            dominated = True
                            break
                    
                    if not dominated:
                        # Update the local best for this polish branch
                        # We don't necessarily update the main population, 
                        # but we record the point (which is done by problem call)
                        # We can update base_X to continue search from here
                        base_X = candidate
                        local_sigma *= 0.8
                    else:
                        local_sigma *= 0.8

        return F
