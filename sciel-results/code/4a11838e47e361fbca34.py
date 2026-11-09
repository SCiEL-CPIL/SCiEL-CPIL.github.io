import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # Budget is 200. Let's use a population size that allows several generations.
        # N = 20, Generations = (200 - 20) / 20 = 9. Total 10 steps.
        # N = 10, Generations = (200 - 10) / 10 = 19. Total 20 steps.
        # Smaller N allows more generations, potentially better convergence, but worse diversity.
        # Let's try N=20.
        N = 20
        if N > self.budget:
            N = self.budget
            
        # Initialize population
        X = np.random.uniform(lo, hi, (N, self.dim))
        F = np.array([problem(x) for x in X])
        used_evals = N
        
        # NSGA-II parameters
        # Crossover probability
        pc = 0.9
        # Crossover distribution index (eta_c)
        eta_c = 15.0
        # Mutation probability
        pm = 1.0 / self.dim
        # Mutation distribution index (eta_m)
        eta_m = 20.0
        
        while used_evals < self.budget:
            # Generate offspring population
            X_off = np.zeros((N, self.dim))
            F_off = np.zeros((N, self.n_obj))
            
            for i in range(N):
                # Select parents using tournament selection based on non-dominated sorting
                # We need to sort the current population first
                # To save time, we can approximate or just use random selection if budget is very tight,
                # but NSGA-II requires sorting. Let's implement a lightweight sort.
                
                # Actually, sorting every generation for small N is cheap.
                # Let's do non-dominated sorting on current X, F
                ranks, crowding = self._nsga2_sort(X, F, N)
                
                # Tournament selection for parent 1
                p1_idx = self._tournament_select(ranks, crowding, N, 2)
                # Tournament selection for parent 2
                p2_idx = self._tournament_select(ranks, crowding, N, 2)
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # Crossover
                if np.random.rand() < pc:
                    c1, c2 = self._sbx_crossover(x1, x2, lo, hi, eta_c)
                else:
                    c1, c2 = x1.copy(), x2.copy()
                
                # Mutation
                c1 = self._poly_mutation(c1, lo, hi, pm, eta_m)
                
                # Evaluate offspring
                f1 = problem(c1)
                used_evals += 1
                if used_evals > self.budget:
                    break
                    
                X_off[i] = c1
                F_off[i] = f1
                
                # If we have budget for the second offspring, we could generate it,
                # but standard NSGA-II generates N offspring from N pairs.
                # To keep it simple and within budget, we generate one offspring per parent pair slot.
                # The loop runs N times, generating N offspring.
                
            # Combine parent and offspring populations
            X_all = np.vstack((X, X_off))
            F_all = np.vstack((F, F_off))
            
            # Non-dominated sort the combined population
            ranks_all, crowding_all = self._nsga2_sort(X_all, F_all, 2 * N)
            
            # Select next generation
            # We want to select N individuals from the 2N
            # Sort by rank, then by crowding distance
            # Create an index array sorted by rank, then crowding
            sort_idx = np.lexsort((crowding_all, ranks_all))
            # We want the first N in this sorted order (lowest rank, highest crowding)
            # Note: lexsort sorts by last key first. So (crowding, ranks) means primary ranks, secondary crowding.
            # We want lowest ranks. For same rank, highest crowding.
            # So we should sort by ranks ascending, then crowding descending.
            # np.lexsort(( -crowding_all, ranks_all ))
            
            sort_idx = np.lexsort((-crowding_all, ranks_all))
            
            # Take first N
            next_idx = sort_idx[:N]
            
            X = X_all[next_idx]
            F = F_all[next_idx]
            
            # Check if we still have budget for the next generation's offspring
            # The loop condition handles this.
            
        return X

    def _nsga2_sort(self, X, F, N):
        """
        Performs non-dominated sorting and calculates crowding distance.
        Returns ranks (array of ints) and crowding distances (array of floats).
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n, dtype=float)
        
        # Non-dominated sorting
        S = [set() for _ in range(n)]  # S[i] = set of indices dominated by i
        n_dominated = np.zeros(n, dtype=int)  # n_dominated[i] = number of individuals dominated by i
        front = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                # i dominates j if F[i] <= F[j] and F[i] != F[j]
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dominated[j] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].add(i)
                    n_dominated[i] += 1
        
        # Front 0
        for i in range(n):
            if n_dominated[i] == 0:
                front.append(i)
                ranks[i] = 0
        
        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        ranks[j] = k + 1
            front = next_front
            k += 1
        
        # Calculate crowding distance for each front
        for k in range(max(ranks) + 1):
            front_indices = np.where(ranks == k)[0]
            if len(front_indices) <= 2:
                crowding[front_indices] = np.inf
                continue
            
            # For each objective
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = front_indices[np.argsort(F[front_indices, m])]
                # Assign infinite distance to best and worst
                crowding[sorted_idx[0]] = np.inf
                crowding[sorted_idx[-1]] = np.inf
                
                # Normalize
                f_min = F[sorted_idx[0], m]
                f_max = F[sorted_idx[-1], m]
                if f_max == f_min:
                    continue
                    
                for i in range(1, len(sorted_idx) - 1):
                    idx = sorted_idx[i]
                    dist = (F[sorted_idx[i+1], m] - F[sorted_idx[i-1], m]) / (f_max - f_min)
                    crowding[idx] += dist
        
        return ranks, crowding

    def _tournament_select(self, ranks, crowding, N, k=2):
        """
        Tournament selection.
        """
        # Select k random individuals
        indices = np.random.choice(N, k, replace=False)
        # Find the best one
        best_idx = 0
        best_rank = ranks[indices[0]]
        best_crowding = crowding[indices[0]]
        
        for i in range(1, k):
            idx = indices[i]
            r = ranks[idx]
            c = crowding[idx]
            # Lower rank is better. If same rank, higher crowding is better.
            if r < best_rank or (r == best_rank and c > best_crowding):
                best_idx = i
                best_rank = r
                best_crowding = c
        
        return indices[best_idx]

    def _sbx_crossover(self, x1, x2, lo, hi, eta_c):
        """
        Simulated Binary Crossover.
        """
        c1 = np.zeros_like(x1)
        c2 = np.zeros_like(x2)
        
        for i in range(len(x1)):
            if np.random.rand() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    if np.random.rand() <= 1.0 / (eta_c + 1):
                        beta = (2.0 * np.random.rand()) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2.0 * np.random.rand())) ** (1.0 / (eta_c + 1))
                    
                    c1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                    c2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                else:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
            else:
                c1[i] = x2[i]
                c2[i] = x1[i]
                
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2

    def _poly_mutation(self, x, lo, hi, pm, eta_m):
        """
        Polynomial Mutation.
        """
        x_mut = x.copy()
        for i in range(len(x)):
            if np.random.rand() < pm:
                if abs(hi[i] - lo[i]) < 1e-14:
                    continue
                    
                delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                
                if np.random.rand() < 0.5:
                    xy = np.random.rand()
                    mut_pow = 2.0 / (eta_m + 1.0) * (1.0 + 2.0 * delta1) ** (eta_m + 1.0)
                    if 2.0 * xy <= mut_pow:
                        beta = (2.0 * xy) ** (1.0 / (eta_m + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - xy))) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] = x_mut[i] - (hi[i] - lo[i]) * (beta - 1.0)
                else:
                    xy = np.random.rand()
                    mut_pow = 2.0 / (eta_m + 1.0) * (1.0 + 2.0 * delta2) ** (eta_m + 1.0)
                    if 2.0 * xy <= mut_pow:
                        beta = (2.0 * xy) ** (1.0 / (eta_m + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - xy))) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] = x_mut[i] + (hi[i] - lo[i]) * (beta - 1.0)
                    
        x_mut = np.clip(x_mut, lo, hi)
        return x_mut
