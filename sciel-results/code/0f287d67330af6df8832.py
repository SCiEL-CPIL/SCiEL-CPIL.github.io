import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.rng = np.random.default_rng(42)

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size and generations
        # Budget = 200. 
        # Let N = 20.
        # Init: 20 evals.
        # Remaining: 180 evals.
        # Each generation produces N offspring.
        # Generations G = 180 / 20 = 9.
        N = 20
        G = (self.budget - N) // N
        if G < 1:
            N = self.budget // 2
            G = 1

        # Initialize Population
        X = lo + (hi - lo) * self.rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        self.budget_used = N

        def fast_non_dominated_sort(F):
            """
            Returns list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_d = np.zeros(n, dtype=int)  # Number of dominators
            rank = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
                        
            for i in range(n):
                if n_d[i] == 0:
                    rank[i] = 0
                    fronts.append([i])
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                    
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for individuals in indices.
            Returns a dict {index: distance}
            """
            m = len(indices)
            if m <= 2:
                return {i: np.inf for i in indices}
            
            cd = {i: 0.0 for i in indices}
            f_sub = F[indices]
            
            for k in range(self.n_obj):
                sorted_indices = sorted(indices, key=lambda i: F[i][k])
                cd[sorted_indices[0]] = np.inf
                cd[sorted_indices[-1]] = np.inf
                
                f_min = F[sorted_indices[0]][k]
                f_max = F[sorted_indices[-1]][k]
                
                if f_max == f_min:
                    continue
                
                for j in range(1, m - 1):
                    cd[sorted_indices[j]] += (F[sorted_indices[j+1]][k] - F[sorted_indices[j-1]][k]) / (f_max - f_min)
                    
            return cd

        def select_parents(F, rank, cd):
            """
            Tournament selection based on rank and crowding distance.
            """
            selected = []
            for _ in range(N):
                i = self.rng.integers(0, N)
                j = self.rng.integers(0, N)
                
                if rank[i] < rank[j]:
                    selected.append(i)
                elif rank[i] > rank[j]:
                    selected.append(j)
                else:
                    if cd[i] > cd[j]:
                        selected.append(i)
                    else:
                        selected.append(j)
            return selected

        def sbx_crossover(x1, x2, lo, hi):
            """
            Simulated Binary Crossover.
            """
            if self.rng.random() > 0.9:
                return x1.copy(), x2.copy()
                
            eta_c = 15.0
            x1_new = np.zeros(self.dim)
            x2_new = np.zeros(self.dim)
            
            for i in range(self.dim):
                if self.rng.random() < 0.5:
                    if x1[i] == x2[i]:
                        x1_new[i] = x1[i]
                        x2_new[i] = x2[i]
                    else:
                        if x1[i] < x2[i]:
                            x1_low, x2_high = x1[i], x2[i]
                        else:
                            x1_low, x2_high = x2[i], x1[i]
                        
                        if (x2_high - x1_low) < 1e-12:
                            x1_new[i] = x1[i]
                            x2_new[i] = x2[i]
                        else:
                            u = self.rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (eta_c + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                                
                            c1 = 0.5 * (x1[i] + x2[i] - beta * (x2[i] - x1[i]))
                            c2 = 0.5 * (x1[i] + x2[i] + beta * (x2[i] - x1[i]))
                            
                            c1 = np.clip(c1, lo[i], hi[i])
                            c2 = np.clip(c2, lo[i], hi[i])
                            
                            x1_new[i] = c1
                            x2_new[i] = c2
                else:
                    x1_new[i] = x1[i]
                    x2_new[i] = x2[i]
                    
            return x1_new, x2_new

        def polynomial_mutation(x, lo, hi):
            """
            Polynomial Mutation.
            """
            eta_m = 20.0
            x_new = x.copy()
            for i in range(self.dim):
                if self.rng.random() < (1.0 / self.dim):
                    u = self.rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    
                    x_new[i] = x[i] + delta * (hi[i] - lo[i])
                    x_new[i] = np.clip(x_new[i], lo[i], hi[i])
            return x_new

        # Main Loop
        for g in range(G):
            if self.budget_used >= self.budget:
                break
                
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # Need to maintain rank and cd for current population
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(N, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
                    
            cd = crowding_distance(F, list(range(N)))
            
            for _ in range(N // 2):
                p1_idx = self.rng.integers(0, N)
                p2_idx = self.rng.integers(0, N)
                
                # Tournament selection for parents
                # Actually, standard NSGA-II uses tournament on the whole population.
                # Let's do 2-tournament selection.
                
                # Parent 1
                i = self.rng.integers(0, N)
                j = self.rng.integers(0, N)
                if rank[i] < rank[j] or (rank[i] == rank[j] and cd[i] >= cd[j]):
                    p1 = i
                else:
                    p1 = j
                    
                # Parent 2
                i = self.rng.integers(0, N)
                j = self.rng.integers(0, N)
                if rank[i] < rank[j] or (rank[i] == rank[j] and cd[i] >= cd[j]):
                    p2 = i
                else:
                    p2 = j
                    
                x1, x2 = sbx_crossover(X[p1], X[p2], lo, hi)
                x1 = polynomial_mutation(x1, lo, hi)
                x2 = polynomial_mutation(x2, lo, hi)
                
                offspring_X.append(x1)
                offspring_X.append(x2)
                
                f1 = problem(x1)
                f2 = problem(x2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                
                self.budget_used += 2
                
                if self.budget_used >= self.budget:
                    break
            
            # If we broke early due to budget, resize offspring to match N?
            # Actually, if we break, we might have less than N offspring.
            # We should handle this carefully.
            # Let's ensure we only create full generations if budget allows.
            # With N=20, G=9, we use exactly 200. So no partial generation expected.
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            total_X = np.vstack([X, offspring_X])
            total_F = np.vstack([F, offspring_F])
            
            # Non-dominated sorting on combined population
            total_fronts = fast_non_dominated_sort(total_F)
            
            # Select next generation
            next_X = []
            next_F = []
            next_rank = np.zeros(2 * N, dtype=int)
            
            for i, front in enumerate(total_fronts):
                if len(next_X) + len(front) <= N:
                    for idx in front:
                        next_X.append(total_X[idx])
                        next_F.append(total_F[idx])
                        next_rank[idx] = i
                else:
                    # Need to select from this front based on crowding distance
                    # Calculate crowding distance for this front
                    front_indices = front
                    front_F = total_F[front_indices]
                    
                    # Map original indices to local indices for cd calculation
                    # Actually, my crowding_distance function takes global indices and F.
                    cd_front = crowding_distance(total_F, front_indices)
                    
                    # Sort front_indices by cd descending
                    sorted_front = sorted(front_indices, key=lambda i: cd_front[i], reverse=True)
                    
                    # Take only the needed number
                    needed = N - len(next_X)
                    for idx in sorted_front[:needed]:
                        next_X.append(total_X[idx])
                        next_F.append(total_F[idx])
                        next_rank[idx] = i
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
