import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 20 # Population size
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1
        self.sbx_eta = 15.0 # Distribution index for SBX
        self.pm_eta = 20.0 # Distribution index for Polynomial Mutation

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initialize Population
        X = lo + (hi - lo) * np.random.random((self.pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        
        used = self.pop_size
        
        # Helper: Fast Non-Dominated Sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]  # S[i] = set of j dominated by i
            n_doms = [0 for _ in range(n)] # n_doms[i] = number of individuals dominating i
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_doms[i] += 1
                if n_doms[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return [float('inf')] * len(front_indices)
            
            distances = [0.0] * len(front_indices)
            m = self.n_obj
            
            for k in range(m):
                # Sort front by k-th objective
                sorted_front = sorted(front_indices, key=lambda i: F[i][k])
                distances[sorted_front[0]] = float('inf')
                distances[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                
                if f_max == f_min:
                    continue
                    
                for i in range(1, len(sorted_front) - 1):
                    idx = sorted_front[i]
                    distances[idx] += (F[sorted_front[i+1]][k] - F[sorted_front[i-1]][k]) / (f_max - f_min)
            
            return distances

        # Helper: Binary Tournament Selection
        def tournament_select(pop_F, pop_X, ranks, cdists):
            i = np.random.randint(len(pop_F))
            j = np.random.randint(len(pop_F))
            
            if ranks[i] < ranks[j]:
                return i
            elif ranks[i] > ranks[j]:
                return j
            else:
                if cdists[i] > cdists[j]:
                    return i
                else:
                    return j

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2):
            if np.random.rand() > self.crossover_prob:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        if np.random.rand() <= 0.5:
                            beta = (2 * np.random.rand()) ** (1.0 / (self.sbx_eta + 1))
                        else:
                            beta = (1.0 / (2 * np.random.rand())) ** (1.0 / (self.sbx_eta + 1))
                        
                        c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        
                        c1[i] = np.clip(c1[i], lo[i], hi[i])
                        c2[i] = np.clip(c2[i], lo[i], hi[i])
            return c1, c2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x):
            for i in range(self.dim):
                if np.random.rand() < self.mutation_prob:
                    delta = np.random.rand()
                    if delta <= 0.5:
                        delta1 = (2 * delta) ** (1.0 / (self.pm_eta + 1)) - 1
                    else:
                        delta1 = 1 - (2 * (1 - delta)) ** (1.0 / (self.pm_eta + 1))
                    
                    x[i] += delta1 * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main Loop
        while used < self.budget:
            # Generate Offspring
            offspring_X = []
            offspring_F = []
            
            # We need to create a new population of size M
            # We can generate pairs until we have M offspring
            # Since budget is tight, we generate as many as possible in this "generation"
            # But standard NSGA-II generates a full population of offspring.
            # If we don't have enough budget for a full population, we can just stop or generate partial.
            # Given budget=200 and M=20, we can do ~10 generations.
            
            # Let's generate a full population of offspring if possible
            # If used + self.pop_size > self.budget, we might run out.
            # We will generate offspring one by one until budget is reached or pop is full.
            
            # To simplify and ensure we don't exceed budget, let's generate offspring in batches
            # or just check budget before each evaluation.
            
            # Let's generate a full population of offspring in this iteration if we have at least M budget left?
            # No, better to just generate offspring one by one and append to a list, then combine.
            # But NSGA-II selection requires a combined population of 2M.
            # So we need M offspring.
            
            if used + self.pop_size <= self.budget:
                # Generate full population of offspring
                for _ in range(self.pop_size // 2): # M/2 pairs -> M offspring
                    p1_idx = tournament_select(F, X, ranks, cdists)
                    p2_idx = tournament_select(F, X, ranks, cdists)
                    
                    c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                    c1 = polynomial_mutation(c1)
                    c2 = polynomial_mutation(c2)
                    
                    offspring_X.append(c1)
                    offspring_F.append(problem(c1))
                    used += 1
                    
                    if used >= self.budget:
                        break
                    
                    offspring_X.append(c2)
                    offspring_F.append(problem(c2))
                    used += 1
                    
                    if used >= self.budget:
                        break
            else:
                # Not enough budget for a full generation, break or do partial?
                # If we break, we return current F.
                # If we do partial, we can't do proper environment selection.
                # Let's just break if we can't do a full generation to keep the algorithm structure clean.
                # However, if we have some budget left, we could do random search or something?
                # The prompt says "More than 200 evaluations... makes the whole result invalid".
                # So we must be careful.
                # Let's just break.
                break
            
            if used >= self.budget:
                break
            
            # Combine Parent and Offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])
            
            # Environment Selection
            fronts = fast_non_dominated_sort(F_combined)
            
            # Assign ranks
            ranks = [0] * len(F_combined)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Select next M individuals
            next_X = []
            next_F = []
            cdists = [0.0] * len(F_combined)
            
            for front in fronts:
                if len(next_X) + len(front) <= self.pop_size:
                    # Add all from this front
                    for idx in front:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                else:
                    # Need to select some from this front based on crowding distance
                    cd = crowding_distance(F_combined, front)
                    # Sort front by crowding distance descending
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = self.pop_size - len(next_X)
                    for idx in front_sorted[:needed]:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
            # Recalculate ranks and cdists for next generation selection
            # We need ranks and cdists for the tournament selection in the next loop
            fronts = fast_non_dominated_sort(F)
            ranks = [0] * len(F)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Calculate crowding distance for the whole population?
            # Tournament selection uses ranks and cdists.
            # Usually cdists are calculated for the fronts.
            # For simplicity in tournament, we can calculate cdists for the entire population
            # or just for the fronts they belong to.
            # Let's calculate cdists for each front and store in an array.
            cdists = [0.0] * len(F)
            for front in fronts:
                cd = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cdists[idx] = cd[i]
        
        return F
