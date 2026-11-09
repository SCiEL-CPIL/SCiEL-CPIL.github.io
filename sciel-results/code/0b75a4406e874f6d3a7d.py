import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
                start_time = time.time()
                deadline = start_time + 290  # Safety margin
                
                rng = np.random  # Use global np.random seeded by evaluator
                
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                span = hi - lo
                
                # Population size
                N = 30 
                
                # Initialize
                X = lo + span * rng.random((N, self.dim))
                F = np.array([problem(x) for x in X])
                used = N
                
                # Helper: Non-dominated sorting
                def fast_non_dominated_sort(F):
                    # Returns list of ranks, where ranks[i] is the rank of individual i
                    n = len(F)
                    S = [[] for _ in range(n)]
                    n_doms = np.zeros(n, dtype=int)
                    ranks = np.zeros(n, dtype=int)
                    
                    for i in range(n):
                        for j in range(n):
                            if i == j: continue
                            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                                S[i].append(j)
                                n_doms[i] += 1
                            elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                n_doms[j] += 1
                                S[j].append(i) # Wait, S[i] should contain j if i dominates j? 
                                # Let's standardize: S[i] = set of j dominated by i.
                                # If i dominates j, j is in S[i].
                                
                    # Correcting logic:
                    S = [[] for _ in range(n)]
                    n_doms = np.zeros(n, dtype=int)
                    for i in range(n):
                        for j in range(n):
                            if i == j: continue
                            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                                S[i].append(j)
                                n_doms[i] += 1
                            elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                n_doms[j] += 1
                                S[j].append(i)
                                
                    rank = 0
                    current_front = [i for i in range(n) if n_doms[i] == 0]
                    ranks = np.full(n, -1, dtype=int)
                    while current_front:
                        next_front = []
                        for i in current_front:
                            ranks[i] = rank
                            for j in S[i]:
                                n_doms[j] -= 1
                                if n_doms[j] == 0:
                                    next_front.append(j)
                        current_front = next_front
                        rank += 1
                    return ranks

                # Helper: Crowding Distance
                def crowding_distance(F, rank_idx):
                    # F is the full array, rank_idx is the list of indices in this rank
                    n = len(rank_idx)
                    if n <= 2:
                        return np.full(n, np.inf)
                    
                    cd = np.zeros(n)
                    m = self.n_obj
                    for k in range(m):
                        # Sort indices by objective k
                        sorted_idx = sorted(range(n), key=lambda i: F[rank_idx[i], k])
                        cd[sorted_idx[0]] = np.inf
                        cd[sorted_idx[-1]] = np.inf
                        f_min = F[rank_idx[sorted_idx[0]], k]
                        f_max = F[rank_idx[sorted_idx[-1]], k]
                        if f_max == f_min:
                            continue
                        for i in range(1, n-1):
                            prev = rank_idx[sorted_idx[i-1]]
                            curr = rank_idx[sorted_idx[i]]
                            next_ = rank_idx[sorted_idx[i+1]]
                            cd[i] += (F[next_, k] - F[prev, k]) / (f_max - f_min)
                    return cd

                # Helper: Prune population to size N
                def prune_population(X, F, N):
                    if len(X) <= N:
                        return X, F
                    
                    ranks = fast_non_dominated_sort(F)
                    # Sort by rank
                    sorted_indices = np.argsort(ranks)
                    # Group by rank
                    unique_ranks, counts = np.unique(ranks, return_counts=True)
                    
                    new_X = []
                    new_F = []
                    remaining = N
                    for rank_val in unique_ranks:
                        count = int(counts[unique_ranks == rank_val][0]) # This is wrong, need mapping
                        # Better:
                        idx_in_rank = [i for i in sorted_indices if ranks[i] == rank_val]
                        
                        if remaining >= count:
                            for i in idx_in_rank:
                                new_X.append(X[i])
                                new_F.append(F[i])
                            remaining -= count
                        else:
                            # Need to select 'remaining' from this rank based on crowding distance
                            cd = crowding_distance(F, idx_in_rank)
                            # Sort by CD descending
                            sorted_by_cd = sorted(range(len(idx_in_rank)), key=lambda i: cd[i], reverse=True)
                            for j in range(remaining):
                                idx_to_keep = idx_in_rank[sorted_by_cd[j]]
                                new_X.append(X[idx_to_keep])
                                new_F.append(F[idx_to_keep])
                            remaining = 0
                            break
                            
                    return np.array(new_X), np.array(new_F)

                # Main Loop
                while used < self.budget:
                    if time.time() > deadline:
                        break
                    
                    # Select parents
                    # Tournament selection on rank and crowding? 
                    # Simpler: Pick 2 random non-dominated points from current population?
                    # Or just pick 2 random points from the current population.
                    # To ensure diversity, we should pick from the current front.
                    # Let's identify the current non-dominated set.
                    ranks = fast_non_dominated_sort(F)
                    non_dom_idx = [i for i in range(len(F)) if ranks[i] == 0]
                    
                    # If we have a large non-dominated set, pick from it.
                    # If small, pick from all.
                    parent_pool = non_dom_idx if len(non_dom_idx) > 1 else list(range(len(F)))
                    
                    p1_idx = rng.choice(parent_pool)
                    p2_idx = rng.choice(parent_pool)
                    
                    # Crossover: BLX-Alpha or SBX
                    # Let's use a simple blend: child = 0.5*(p1+p2) + noise
                    # Or SBX. Let's use SBX for better distribution.
                    # eta_c = 15
                    def sbx_crossover(x1, x2, eta=15.0):
                        x1 = np.array(x1)
                        x2 = np.array(x2)
                        x3 = np.zeros_like(x1)
                        x4 = np.zeros_like(x1)
                        for i in range(len(x1)):
                            if rng.random() < 0.5:
                                if abs(x2[i] - x1[i]) < 1e-14:
                                    x3[i] = x1[i]
                                    x4[i] = x2[i]
                                else:
                                    u = rng.random()
                                    if u <= 0.5:
                                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                                    else:
                                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                                    
                                    c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                                    c2 = 0.5 * ((1 + 1/beta) * x1[i] + (1 - 1/beta) * x2[i])
                                    
                                    # Clip to bounds
                                    c1 = np.clip(c1, lo[i], hi[i])
                                    c2 = np.clip(c2, lo[i], hi[i])
                                    
                                    x3[i] = c1
                                    x4[i] = c2
                            else:
                                x3[i] = x1[i]
                                x4[i] = x2[i]
                        return x3, x4

                    child1, child2 = sbx_crossover(X[p1_idx], X[p2_idx], eta=15.0)
                    
                    # Mutation: Polynomial or Gaussian
                    # Let's use Gaussian with sigma = 0.1 * span
                    def mutate(x, sigma=0.1):
                        x = np.array(x)
                        x += sigma * span * rng.standard_normal(self.dim)
                        return np.clip(x, lo, hi)

                    child1 = mutate(child1)
                    child2 = mutate(child2)
                    
                    # Evaluate
                    f1 = problem(child1)
                    used += 1
                    f2 = problem(child2)
                    used += 1
                    
                    # Add to population
                    X = np.vstack([X, child1, child2])
                    F = np.vstack([F, f1, f2])
                    
                    # Prune
                    X, F = prune_population(X, F, N)
                    
                return X
