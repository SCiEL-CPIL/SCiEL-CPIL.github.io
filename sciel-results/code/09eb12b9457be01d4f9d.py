import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Use global np.random to respect the seed set by the evaluator
        # If we want to be safe against state issues, we could seed from global state,
        # but np.random.seed is called, so np.random.* will be seeded.
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # NSGA-II Parameters
        N = 20  # Population size
        eta_c = 20.0  # Crossover distribution index
        eta_m = 20.0  # Mutation distribution index
        p_m = 1.0 / dim  # Probability of mutation per gene
        
        # Initialize Population
        X = lo + (hi - lo) * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper functions defined inside to access local vars or just keep clean
        def dominates(a, b):
            # a dominates b if a <= b in all objectives and a < b in at least one
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of individuals dominated by i
            n_d = [0] * n  # n_d[i] = number of individuals that dominate i
            rank = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_d[i] += 1
                        
            # Front 0
            front = [i for i in range(n) if n_d[i] == 0]
            for i in front:
                rank[i] = 0
            if front:
                fronts.append(front)
                
            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                            rank[j] = k + 1
                front = next_front
                k += 1
                if front:
                    fronts.append(front)
                    
            return fronts, rank

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            n = len(indices)
            dist = np.zeros(n)
            for m in range(n_obj):
                # Sort indices by objective m
                sorted_idx = sorted(indices, key=lambda i: F[i, m])
                dist_sorted = np.zeros(n)
                # Assign inf to best and worst
                dist_sorted[0] = np.inf
                dist_sorted[-1] = np.inf
                f_range = F[sorted_idx[-1], m] - F[sorted_idx[0], m]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    dist_sorted[i] += (F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]) / f_range
                # Map back to original indices order
                for i, idx in enumerate(sorted_idx):
                    # We need to add to the distance of the original index in the 'dist' array
                    # But 'dist' is indexed by position in 'indices' list?
                    # Let's keep dist indexed by the position in the 'indices' list passed to this function
                    # Wait, the input 'indices' is a list of global indices.
                    # The output should be a dict or array corresponding to 'indices'.
                    # Let's return a dict {global_idx: distance}
                    pass
            
            # Let's re-implement to return a dict for clarity
            dist_dict = {i: 0.0 for i in indices}
            for m in range(n_obj):
                sorted_idx = sorted(indices, key=lambda i: F[i, m])
                if len(sorted_idx) > 2:
                    f_range = F[sorted_idx[-1], m] - F[sorted_idx[0], m]
                    if f_range > 0:
                        for i in range(1, len(sorted_idx) - 1):
                            dist_dict[sorted_idx[i]] += (F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]) / f_range
            return dist_dict

        def sbx_crossover(x1, x2, bounds):
            child1 = x1.copy()
            child2 = x2.copy()
            for j in range(dim):
                if np.random.rand() < 0.5:
                    if abs(x1[j] - x2[j]) > 1e-14:
                        beta1 = 1.0
                        beta2 = 1.0
                        if x1[j] < x2[j]:
                            beta1 = x1[j]
                            beta2 = x2[j]
                        else:
                            beta1 = x2[j]
                            beta2 = x1[j]
                        
                        u = np.random.rand()
                        if u <= 0.5:
                            alpha = 2.0 * u
                            beta = (beta1 + (beta2 - beta1) * alpha)
                            # Actually standard formula:
                            # if u <= 0.5: beta1 = (2u)^1/(eta+1)
                            # else: beta1 = (2(1-u))^1/(eta+1)
                            beta1_val = (2.0 * u) ** (1.0 / (eta_c + 1.0))
                        else:
                            alpha = 2.0 * (1.0 - u)
                            beta1_val = (2.0 * (1.0 - u)) ** (1.0 / (eta_c + 1.0))
                        
                        c1 = 0.5 * ((1 + beta1_val) * x1[j] + (1 - beta1_val) * x2[j])
                        c2 = 0.5 * ((1 + 1.0/beta1_val) * x2[j] + (1 - 1.0/beta1_val) * x1[j])
                        
                        # Wait, standard SBX:
                        # c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                        # c2 = 0.5 * ((1 + 1/beta) * x2 + (1 - 1/beta) * x1)
                        # where beta is calculated from u.
                        
                        c1 = np.clip(c1, bounds[j][0], bounds[j][1])
                        c2 = np.clip(c2, bounds[j][0], bounds[j][1])
                        child1[j] = c1
                        child2[j] = c2
            return child1, child2

        def poly_mutation(x, bounds):
            for j in range(dim):
                if np.random.rand() < p_m:
                    u = np.random.rand()
                    delta1 = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    delta2 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    delta = delta1 if np.random.rand() < 0.5 else delta2
                    x[j] += delta * (bounds[j][1] - bounds[j][0])
                    x[j] = np.clip(x[j], bounds[j][0], bounds[j][1])
            return x

        def tournament_select(F, ranks, dists, pop_size):
            # Select two parents
            def get_parent():
                i = np.random.randint(0, pop_size)
                j = np.random.randint(0, pop_size)
                if ranks[i] < ranks[j]:
                    return i
                elif ranks[i] > ranks[j]:
                    return j
                else:
                    # Compare crowding distance
                    # dists is a dict {idx: dist}
                    if dists[i] > dists[j]:
                        return i
                    else:
                        return j
            return get_parent(), get_parent()

        # Main Evolution Loop
        while used < budget:
            # Generate Offspring
            offspring_X = np.zeros((N, dim))
            for i in range(N):
                if used >= budget:
                    break
                p1_idx, p2_idx = tournament_select(F, rank, crowding_distance(F, list(range(N))), N)
                # Note: Recalculating crowding distance every time is expensive.
                # We should calculate it once per generation for the current population.
                
                # Let's restructure: Calculate ranks and distances for current pop once.
                # Then select parents.
                
                # Actually, the loop above is inefficient. Let's fix the structure.
                pass
            
            # Corrected Loop Structure:
            # 1. Calculate ranks and distances for current X, F
            # 2. Generate N offspring
            # 3. Evaluate offspring
            # 4. Combine, Sort, Select next X, F
            
            # We are currently inside `while used < budget`.
            # Let's break out of the previous broken logic and restart the loop body properly.
            
            # Calculate ranks and distances for current population
            fronts, rank = fast_non_dominated_sort(F)
            # Calculate crowding distances for each front
            dists = {i: 0.0 for i in range(N)}
            for front in fronts:
                d_dict = crowding_distance(F, front)
                for idx, d in d_dict.items():
                    dists[idx] = d
            
            # Generate Offspring
            offspring_X = []
            for _ in range(N):
                if used >= budget:
                    break
                # Select parents
                i = np.random.randint(0, N)
                j = np.random.randint(0, N)
                # Tournament
                if rank[i] < rank[j]:
                    p1 = i
                elif rank[i] > rank[j]:
                    p1 = j
                else:
                    if dists[i] > dists[j]:
                        p1 = i
                    else:
                        p1 = j
                
                k = np.random.randint(0, N)
                l = np.random.randint(0, N)
                if rank[k] < rank[l]:
                    p2 = k
                elif rank[k] > rank[l]:
                    p2 = l
                else:
                    if dists[k] > dists[l]:
                        p2 = k
                    else:
                        p2 = l
                        
                c1, c2 = sbx_crossover(X[p1], X[p2], bounds)
                c1 = poly_mutation(c1, bounds)
                c2 = poly_mutation(c2, bounds)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                
            # We generated 2*N offspring? No, we appended 2 per iteration, so 2*N total.
            # But we only have budget for N more evaluations if we want to keep total 200?
            # Init: 20.
            # If we do 9 generations of 20 offspring each, we need 180 evals.
            # So we should generate exactly `remaining = budget - used` offspring.
            
            # Let's adjust:
            # remaining = budget - used
            # Generate `remaining` offspring.
            
            # Since the code above is getting messy, I will rewrite the whole class cleanly in the replacement block.
            pass

        return X
