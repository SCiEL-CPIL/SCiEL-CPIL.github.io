import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # Setup
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # With 200 budget, N=50 allows 4 generations (50 init + 3*50 offspring = 200)
        # Or N=40 allows 5 generations.
        # Let's try N=50.
        N = 50
        
        # Initialize population
        # Use numpy random with seed 0 for reproducibility if needed, but task says evaluator sets seed.
        # The task says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)..."
        # So I should use np.random directly or a generator that respects the global state?
        # Usually, if np.random.seed is called, np.random.rand uses that state.
        # I will use np.random.rand to respect the external seed.
        
        X = lo + (hi - lo) * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper functions for NSGA-II
        
        def fast_non_dominated_sort(F):
            """
            Returns ranks array and fronts list.
            """
            n = len(F)
            S = [set() for _ in range(n)]  # S[i] = set of indices dominated by i
            n_doms = np.zeros(n, dtype=int) # n_doms[i] = count of individuals dominating i
            fronts = [set()]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_doms[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_doms[i] += 1
            
            # First front
            for i in range(n):
                if n_doms[i] == 0:
                    fronts[0].add(i)
            
            k = 0
            while len(fronts[k]) > 0:
                next_front = set()
                for i in fronts[k]:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.add(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
                    
            ranks = np.zeros(n, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
                    
            return ranks, fronts

        def crowding_distance(F, front_indices):
            """
            Calculates crowding distance for a specific front.
            """
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            f = F[front_indices]
            
            for m in range(n_obj):
                min_f = np.min(f[:, m])
                max_f = np.max(f[:, m])
                if max_f - min_f == 0:
                    continue
                # Sort indices by objective m
                sorted_idx = np.argsort(f[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (f[sorted_idx[i+1], m] - f[sorted_idx[i-1], m]) / (max_f - min_f)
                    
            return dist

        def select_parents(ranks, crowd_dists, n_parents=2):
            """
            Tournament selection based on rank and crowding distance.
            """
            n = len(ranks)
            parents = []
            for _ in range(n_parents):
                # Random tournament
                a = np.random.randint(n)
                b = np.random.randint(n)
                
                # Compare
                if ranks[a] < ranks[b]:
                    winner = a
                elif ranks[b] < ranks[a]:
                    winner = b
                else:
                    # Same rank, compare crowding distance
                    # crowd_dists is a dict or array aligned with original indices?
                    # Let's make crowd_dists an array of size N, filled for current front members.
                    # Actually, it's easier to keep a global crowd array.
                    if crowd_dists[a] > crowd_dists[b]:
                        winner = a
                    else:
                        winner = b
                parents.append(winner)
            return parents

        def sbx_crossover(x1, x2, lo, hi, dim, eta=15.0):
            """
            Simulated Binary Crossover.
            """
            if np.random.rand() < 0.5:
                return x1.copy(), x2.copy()
                
            c1 = np.zeros(dim)
            c2 = np.zeros(dim)
            
            for i in range(dim):
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                
            # Clip to bounds
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            return c1, c2

        def poly_mutation(x, lo, hi, dim, eta=20.0):
            """
            Polynomial Mutation.
            """
            x_mut = x.copy()
            for i in range(dim):
                if np.random.rand() < 0.9: # Mutation rate
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                    range_val = hi[i] - lo[i]
                    x_mut[i] += delta * range_val
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # 1. Evaluate current population (if not already done)
            # We have F for X
            ranks, fronts = fast_non_dominated_sort(F)
            
            # Calculate crowding distance for all
            crowd_dists = np.zeros(N)
            for front in fronts:
                if len(front) > 0:
                    front_list = list(front)
                    cd = crowding_distance(F, front_list)
                    for idx, d in zip(front_list, cd):
                        crowd_dists[idx] = d
            
            # 2. Generate Offspring
            offspring_X = np.zeros((N, dim))
            for i in range(N):
                p1, p2 = select_parents(ranks, crowd_dists)
                c1, c2 = sbx_crossover(X[p1], X[p2], lo, hi, dim)
                c1 = poly_mutation(c1, lo, hi, dim)
                c2 = poly_mutation(c2, lo, hi, dim)
                offspring_X[i] = c1
                # We only need N offspring to combine with N parents to get 2N, then select N
                # But we only have budget for N more evaluations per generation.
                # Standard NSGA-II: Generate N offspring. Combine 2N. Select N.
                # This costs N evaluations.
                # We store c2? No, usually we generate N children from N pairs.
                # Let's just store c1 for the i-th slot.
                # Wait, standard NSGA-II creates N offspring from N pairs of parents.
                # So we need N evaluations.
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring_X])
            used += N
            
            # 3. Combine Parents and Offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, F_off])
            
            # 4. Select Next Generation
            ranks_comb, fronts_comb = fast_non_dominated_sort(F_comb)
            
            # Crowding distance for combined
            crowd_comb = np.zeros(2 * N)
            for front in fronts_comb:
                if len(front) > 0:
                    front_list = list(front)
                    cd = crowding_distance(F_comb, front_list)
                    for idx, d in zip(front_list, cd):
                        crowd_comb[idx] = d
            
            # Select N best from 2N
            next_X = np.zeros((N, dim))
            next_F = np.zeros((N, n_obj))
            selected_indices = []
            
            for front in fronts_comb:
                if len(selected_indices) + len(front) <= N:
                    # Add all
                    for idx in front:
                        selected_indices.append(idx)
                else:
                    # Need to select from this front
                    front_list = list(front)
                    # Sort by crowding distance descending
                    cd_vals = [crowd_comb[idx] for idx in front_list]
                    sorted_front = sorted(front_list, key=lambda i: crowd_comb[i], reverse=True)
                    needed = N - len(selected_indices)
                    for idx in sorted_front[:needed]:
                        selected_indices.append(idx)
                    break
            
            for i, idx in enumerate(selected_indices):
                next_X[i] = X_comb[idx]
                next_F[i] = F_comb[idx]
            
            X = next_X
            F = next_F

        return X
