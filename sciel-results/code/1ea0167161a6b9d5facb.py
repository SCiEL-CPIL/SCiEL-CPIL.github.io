"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 250  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        N = 20
        if N > budget:
            N = budget
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Track all non-dominated points found so far to maximize hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        def fast_non_dominated_sort(F):
            """
            Returns ranks (0 is best) for each point in F.
            """
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1
                        S[j].append(i)
                        
            # Wait, the logic above for S is slightly off. S[i] should contain indices dominated by i.
            # Let's restart the sort logic correctly.
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        
            # This double loop is O(N^2) which is fine for N=20 or 40.
            # Let's do it properly:
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1 # i dominates j, so i's domination count? No.
                        # Standard definition: n_dom[i] is number of points dominating i.
                        # If i dominates j, then n_dom[j] += 1.
                        pass
            
            # Correct implementation:
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            current_front = [i for i in range(n) if n_dom[i] == 0]
            ranks = np.full(n, -1, dtype=int)
            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    ranks[i] = k
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1
            return ranks

        def crowding_distance(F, ranks):
            """
            Calculate crowding distance for points in the same rank.
            """
            n = len(F)
            cd = np.zeros(n)
            unique_ranks = np.unique(ranks)
            
            for r in unique_ranks:
                idx = np.where(ranks == r)[0]
                if len(idx) <= 2:
                    cd[idx] = np.inf
                    continue
                
                # Normalize objectives for this front? 
                # Standard NSGA-II uses raw values or normalized. 
                # For WFG9, scales are similar (1, 1, 1 roughly). 
                # Let's use raw differences.
                for m in range(n_obj):
                    f_min = F[idx, m].min()
                    f_max = F[idx, m].max()
                    if f_max - f_min == 0:
                        continue
                    # Sort indices by m-th objective
                    sorted_idx = idx[np.argsort(F[idx, m])]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    for i in range(1, len(sorted_idx) - 1):
                        prev_idx = sorted_idx[i-1]
                        next_idx = sorted_idx[i+1]
                        cur_idx = sorted_idx[i]
                        cd[cur_idx] += (F[next_idx, m] - F[prev_idx, m]) / (f_max - f_min)
            return cd

        def tournament_select(X, F, ranks, cds, k=2):
            """Select k individuals, return the best one."""
            indices = np.random.choice(len(X), k, replace=False)
            best_idx = indices[0]
            best_rank = ranks[best_idx]
            best_cd = cds[best_idx]
            for idx in indices[1:]:
                if ranks[idx] < best_rank:
                    best_idx = idx
                    best_rank = ranks[idx]
                    best_cd = cds[idx]
                elif ranks[idx] == best_rank and cds[idx] > best_cd:
                    best_idx = idx
                    best_cd = cds[idx]
            return X[best_idx].copy()

        def sbx_crossover(x1, x2, bounds, eta=15.0):
            """Simulated Binary Crossover."""
            x1 = x1.copy()
            x2 = x2.copy()
            # Random mask for which dimensions to crossover
            mask = np.random.random(dim) < 0.5
            for i in range(dim):
                if not mask[i]:
                    continue
                lo_i = bounds[i][0]
                hi_i = bounds[i][1]
                if x1[i] == x2[i]:
                    continue
                # Ensure x1 <= x2 for calculation? 
                # Standard formula handles it if we take min/max
                if x1[i] > x2[i]:
                    x1[i], x2[i] = x2[i], x1[i]
                
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                c1 = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                c2 = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                
                # Clip to bounds
                c1 = np.clip(c1, lo_i, hi_i)
                c2 = np.clip(c2, lo_i, hi_i)
                
                x1[i], x2[i] = c1, c2
            return x1, x2

        def polynomial_mutation(x, bounds, eta=20.0):
            """Polynomial Mutation."""
            x = x.copy()
            for i in range(dim):
                if np.random.random() < 0.5: # Mutation probability 1/dim is standard, but 0.5 is aggressive. Let's use 1/dim.
                    # Actually, standard is 1/dim. Let's stick to 1/dim for stability.
                    pass 
            # Let's do 1/dim probability
            mut_prob = 1.0 / dim
            for i in range(dim):
                if np.random.random() < mut_prob:
                    lo_i = bounds[i][0]
                    hi_i = bounds[i][1]
                    if hi_i == lo_i:
                        continue
                    u = np.random.random()
                    if u < 0.5:
                        delta_1 = (2.0 * u + 1.0 / (hi_i - lo_i)) ** (1.0 / (eta + 1.0)) - 1.0
                        x[i] = x[i] + delta_1 * (hi_i - x[i])
                    else:
                        delta_2 = (2.0 * (1.0 - u) + 1.0 / (hi_i - lo_i)) ** (1.0 / (eta + 1.0)) - 1.0
                        x[i] = x[i] - delta_2 * (x[i] - lo_i)
                    x[i] = np.clip(x[i], lo_i, hi_i)
            return x

        # Main Evolution Loop
        while used < budget:
            if time.time() - start_time > timeout:
                break
                
            # Calculate ranks and crowding distance for current population
            ranks = fast_non_dominated_sort(F)
            cds = crowding_distance(F, ranks)
            
            # Generate offspring
            offspring_X = []
            for _ in range(N):
                p1 = tournament_select(X, F, ranks, cds)
                p2 = tournament_select(X, F, ranks, cds)
                c1, c2 = sbx_crossover(p1, p2, bounds)
                c1 = polynomial_mutation(c1, bounds)
                c2 = polynomial_mutation(c2, bounds)
                offspring_X.append(c1)
                offspring_X.append(c2)
                used += 2
                if used >= budget:
                    break
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            # Evaluate only as many as we have budget for
            # We might have generated more than budget allows if we broke mid-loop?
            # The loop adds 2 per iter. If budget is odd, we stop when used >= budget.
            # So offspring_X length matches (budget - used_before_loop)
            
            offspring_F = np.array([problem(x) for x in offspring_X])
            
            # Combine parent and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Select next generation
            ranks_comb = fast_non_dominated_sort(F_combined)
            cds_comb = crowding_distance(F_combined, ranks_comb)
            
            # Sort indices by rank, then by crowding distance (descending)
            # We need to pick the best N points.
            # Create a list of (rank, -cd, index)
            sort_keys = [(ranks_comb[i], -cds_comb[i], i) for i in range(len(X_combined))]
            sort_keys.sort()
            
            next_X_indices = [idx for _, _, idx in sort_keys[:N]]
            X = X_combined[next_X_indices]
            F = F_combined[next_X_indices]
            
            # Update global non-dominated set
            all_X.append(X)
            all_F.append(F)
            
        # Flatten all points and find non-dominated ones
        all_X_flat = np.vstack(all_X)
        all_F_flat = np.vstack(all_F)
        
        # Find non-dominated points in the entire history
        n_total = len(all_F_flat)
        non_dom_indices = []
        for i in range(n_total):
            is_non_dom = True
            # Optimization: check against current non-dominated set first? 
            # For small budgets (200), O(N^2) check is fine (200*200 = 40000 ops)
            for j in range(n_total):
                if i == j: continue
                if np.all(all_F_flat[j] <= all_F_flat[i]) and np.any(all_F_flat[j] < all_F_flat[i]):
                    is_non_dom = False
                    break
            if is_non_dom:
                non_dom_indices.append(i)
                
        final_F = all_F_flat[non_dom_indices]
        final_X = all_X_flat[non_dom_indices]
        
        return final_X

# EVOLVE-BLOCK-END
