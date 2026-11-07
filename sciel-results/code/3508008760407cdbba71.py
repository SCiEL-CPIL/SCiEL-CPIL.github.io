import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 specific knowledge: x1 should be 0 to minimize f2
        # We can bias initialization towards x1=0
        # But let's stick to a robust NSGA-II implementation first.
        
        pop_size = 50
        if self.budget < pop_size:
            pop_size = self.budget
            
        # Initialize population
        # For ZDT3, x1 is the main driver of f2.
        # Random uniform is fine, but maybe bias towards 0?
        # Let's just use uniform for now, NSGA-II should find it.
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Fast Non-Dominated Sort
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            n_dominates = np.zeros(n, dtype=int)
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dominates[p] += 1
            
            for p in range(n):
                if n_dominates[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dominates[q] -= 1
                        if n_dominates[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            
            if not fronts[-1]:
                fronts.pop()
                
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            dist = np.zeros(len(indices))
            m = self.n_obj
            
            for k in range(m):
                sorted_idx = sorted(indices, key=lambda i: F[i, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                f_min = F[sorted_idx[0], k]
                f_max = F[sorted_idx[-1], k]
                if f_max == f_min:
                    continue
                
                for j in range(1, len(sorted_idx) - 1):
                    idx_j = sorted_idx[j]
                    idx_jm1 = sorted_idx[j-1]
                    idx_jp1 = sorted_idx[j+1]
                    dist[idx_j] += (F[idx_jp1, k] - F[idx_jm1, k]) / (f_max - f_min)
            
            return dist

        # Helper: Tournament Selection with Crowding Distance
        def tournament_select(F, fronts, cd, k=2):
            # Precompute rank for each index for O(1) lookup
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            candidates = [np.random.randint(len(F)) for _ in range(k)]
            best_idx = candidates[0]
            best_rank = rank[best_idx]
            best_cd = cd[best_idx]
            
            for idx in candidates[1:]:
                if rank[idx] < best_rank:
                    best_idx = idx
                    best_rank = rank[idx]
                    best_cd = cd[idx]
                elif rank[idx] == best_rank:
                    if cd[idx] > best_cd:
                        best_idx = idx
                        best_cd = cd[idx]
            
            return best_idx

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Precompute fronts and crowding distance for current population
            fronts = fast_non_dominated_sort(F)
            cd = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 2:
                    cd_front = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cd_front[i]
                elif len(front) == 2:
                    cd[front[0]] = np.inf
                    cd[front[1]] = np.inf
                else:
                    cd[front[0]] = np.inf

            # Generate offspring
            offspring_X = []
            for _ in range(pop_size):
                p1_idx = tournament_select(F, fronts, cd)
                p2_idx = tournament_select(F, fronts, cd)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover: SBX (Simulated Binary Crossover)
                # Vectorized SBX for speed
                r = np.random.random(self.dim)
                beta = np.ones(self.dim)
                mask = r < 0.5
                beta[mask] = (2 * r[mask]) ** (1.0 / (10 + 1))
                beta[~mask] = (1.0 / (2 * r[~mask])) ** (1.0 / (10 + 1))
                
                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                
                # Swap
                swap_mask = np.random.random(self.dim) < 0.5
                child = np.where(swap_mask, c1, c2)
                
                # Mutation: Polynomial
                # Only mutate some genes
                mut_mask = np.random.random(self.dim) < (1.0 / self.dim)
                if np.any(mut_mask):
                    # Vectorized polynomial mutation
                    # beta calculation
                    u = np.random.random(np.sum(mut_mask))
                    beta_mut = np.ones(np.sum(mut_mask))
                    # Check boundaries
                    vals = child[mut_mask]
                    lo_vals = lo[mut_mask]
                    hi_vals = hi[mut_mask]
                    
                    # If at boundary, push inward
                    at_lo = (vals <= lo_vals)
                    at_hi = (vals >= hi_vals)
                    
                    # For interior points
                    interior = ~(at_lo | at_hi)
                    if np.any(interior):
                        u_int = u[interior]
                        val_int = vals[interior]
                        lo_int = lo_vals[interior]
                        hi_int = hi_vals[interior]
                        
                        beta_int = np.ones(np.sum(interior))
                        mask_int = u_int < 0.5
                        beta_int[mask_int] = (2 * (val_int[mask_int] - lo_int[mask_int]) / (hi_int[mask_int] - lo_int[mask_int])) ** (1.0 / (20 + 1))
                        beta_int[~mask_int] = (2 * (hi_int[~mask_int] - val_int[~mask_int]) / (hi_int[~mask_int] - lo_int[~mask_int])) ** (1.0 / (20 + 1))
                        
                        delta = (beta_int - 1) * (hi_int - lo_int)
                        new_vals = val_int + delta
                        new_vals = np.clip(new_vals, lo_int, hi_int)
                        
                        # Update child
                        mut_indices = np.where(mut_mask)[0]
                        int_indices = mut_indices[interior]
                        child[int_indices] = new_vals
                    
                    # For boundary points
                    if np.any(at_lo):
                        mut_indices = np.where(mut_mask)[0]
                        lo_indices = mut_indices[at_lo]
                        child[lo_indices] = lo[lo_indices] + (hi[lo_indices] - lo[lo_indices]) * np.random.random(len(lo_indices))
                    
                    if np.any(at_hi):
                        mut_indices = np.where(mut_mask)[0]
                        hi_indices = mut_indices[at_hi]
                        child[hi_indices] = hi[hi_indices] - (hi[hi_indices] - lo[hi_indices]) * np.random.random(len(hi_indices))
                        
                child = np.clip(child, lo, hi)
                offspring_X.append(child)
            
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)
            
            # Combine
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            # Selection: NSGA-II
            fronts_all = fast_non_dominated_sort(F_all)
            
            new_indices = []
            for front in fronts_all:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    remaining = pop_size - len(new_indices)
                    cd_all = np.zeros(len(F_all))
                    cd_front = crowding_distance(F_all, front)
                    for i, idx in enumerate(front):
                        cd_all[idx] = cd_front[i]
                    sorted_front = sorted(front, key=lambda i: cd_all[i], reverse=True)
                    new_indices.extend(sorted_front[:remaining])
                    break
            
            X = X_all[new_indices]
            F = F_all[new_indices]
        
        return X
