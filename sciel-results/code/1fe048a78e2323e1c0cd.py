import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size
        N = 20
        # Initial population
        X = lo + np.random.rand(N, self.dim) * span
        F = np.array([problem(x) for x in X])
        used = N
        
        # NSGA-II Parameters
        eta_c = 15.0
        eta_m = 20.0
        p_c = 1.0
        p_m = 1.0 / self.dim
        
        # Non-dominated sorting and crowding distance functions
        def fast_non_dominated_sort(F):
            M = len(F)
            S = [[] for _ in range(M)]
            n = [0] * M
            rank = [0] * M
            front = [0]
            i = 0
            for p in range(M):
                for q in range(M):
                    if p == q: continue
                    if F[p,0] <= F[q,0] and F[p,1] <= F[q,1] and F[p,2] <= F[q,2] and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif F[q,0] <= F[p,0] and F[q,1] <= F[p,1] and F[q,2] <= F[p,2] and np.any(F[q] < F[p]):
                        n[p] += 1
                if n[p] == 0:
                    rank[p] = 0
                    front[0] += 1
            while front[i] > 0:
                Q = []
                for p in range(M):
                    if rank[p] == i:
                        Q.append(p)
                        for q in S[p]:
                            n[q] -= 1
                            if n[q] == 0:
                                rank[q] = i + 1
                                front[i+1] += 1 if i+1 < len(front) else 0 # Need dynamic front size
                i += 1
            # Return ranks
            return rank
            
        # Actually, a simpler vectorized approach for small N is better for speed and correctness
        def get_ranks(F):
            M = len(F)
            ranks = np.zeros(M, dtype=int)
            dominated_by = [set() for _ in range(M)]
            dominates = [set() for _ in range(M)]
            
            for i in range(M):
                for j in range(M):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_by[j].add(i)
                        dominates[i].add(j)
            
            current_front = [i for i in range(M) if len(dominated_by[i]) == 0]
            rank = 0
            while current_front:
                for i in current_front:
                    ranks[i] = rank
                next_front = []
                for i in current_front:
                    for j in dominates[i]:
                        dominated_by[j].remove(i)
                        if len(dominated_by[j]) == 0:
                            next_front.append(j)
                current_front = next_front
                rank += 1
            return ranks

        def crowding_distance(F, ranks):
            M = len(F)
            dist = np.zeros(M)
            unique_ranks = np.unique(ranks)
            for r in unique_ranks:
                front_indices = np.where(ranks == r)[0]
                if len(front_indices) <= 2:
                    dist[front_indices] = np.inf
                    continue
                for k in range(self.n_obj):
                    sorted_idx = front_indices[np.argsort(F[front_indices, k])]
                    dist[sorted_idx[0]] = np.inf
                    dist[sorted_idx[-1]] = np.inf
                    min_val = F[sorted_idx[0], k]
                    max_val = F[sorted_idx[-1], k]
                    if max_val - min_val == 0:
                        continue
                    for i in range(1, len(sorted_idx) - 1):
                        idx = sorted_idx[i]
                        prev_idx = sorted_idx[i-1]
                        next_idx = sorted_idx[i+1]
                        dist[idx] += (F[next_idx, k] - F[prev_idx, k]) / (max_val - min_val)
            return dist

        def select_parents(ranks, dist, n_sel):
            # Tournament selection
            parents = []
            for _ in range(n_sel):
                # Pick 2 random
                i = np.random.randint(0, len(ranks))
                j = np.random.randint(0, len(ranks))
                if ranks[i] < ranks[j]:
                    parents.append(i)
                elif ranks[j] < ranks[i]:
                    parents.append(j)
                else:
                    if dist[i] > dist[j]:
                        parents.append(i)
                    else:
                        parents.append(j)
            return parents

        def sbx_crossover(x1, x2):
            if np.random.rand() > p_c:
                return x1.copy(), x2.copy()
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    # Standard SBX
                    if x1[i] == x2[i]:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta_c + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                        c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        child1[i] = c1
                        child2[i] = c2
                else:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
            return child1, child2

        def polynomial_mutation(x):
            for i in range(self.dim):
                if np.random.rand() < p_m:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    x[i] += delta * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # We want to generate enough to fill the next generation
            # Current N=20. Next gen N=20.
            # We have budget for (self.budget - used) evals.
            # Let's generate min(N, self.budget - used) offspring.
            n_off = min(N, self.budget - used)
            
            # Select parents
            ranks = get_ranks(F)
            dist = crowding_distance(F, ranks)
            parents = select_parents(ranks, dist, 2 * n_off)
            
            for i in range(0, 2 * n_off, 2):
                p1 = parents[i]
                p2 = parents[i+1]
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)
                
            # Evaluate only as many as we have budget for
            # Note: The loop above creates 2*n_off children. We evaluate them one by one?
            # No, we can batch evaluate if we want, but problem(x) is a black box.
            # We must ensure we don't exceed budget.
            
            # Actually, simpler: just loop and add until budget is hit.
            # But we need pairs for crossover.
            
            # Let's refine the loop structure.
            # We have `used` evals so far.
            # We can do `remaining = self.budget - used` evals.
            # We will generate children in pairs.
            # Max pairs = remaining // 2.
            
            n_pairs = min(N // 2, (self.budget - used) // 2)
            
            for _ in range(n_pairs):
                # Select 2 parents
                # Re-rank every generation is expensive but N is small.
                # We already ranked F.
                p_idx1, p_idx2 = select_parents(ranks, dist, 2)
                c1, c2 = sbx_crossover(X[p_idx1], X[p_idx2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                f1 = problem(c1)
                used += 1
                f2 = problem(c2)
                used += 1
                
                # Add to offspring pool
                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)
                
                if used >= self.budget:
                    break
            
            if not offspring_X:
                break
                
            # Combine parents and offspring
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.vstack([F, np.array(offspring_F)])
            
            # Select next generation
            ranks_comb = get_ranks(F_comb)
            dist_comb = crowding_distance(F_comb, ranks_comb)
            
            # Sort by rank, then by crowding distance (descending)
            # We need to pick the best N points.
            # Create a list of indices sorted by (rank, -dist)
            indices = np.arange(len(F_comb))
            # Custom sort: primary rank, secondary -dist
            # np.lexsort keys are applied in reverse order.
            # So key 1 is -dist, key 0 is rank.
            sort_keys = np.lexsort((-dist_comb, ranks_comb))
            
            # Take first N
            next_indices = sort_keys[:N]
            
            X = X_comb[next_indices]
            F = F_comb[next_indices]
            
            # Clean up offspring lists
            offspring_X = []
            offspring_F = []
            
        return X
