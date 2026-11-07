import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Setup
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Determine population size
        # We want to maximize generations.
        # Total evals = P + G * P <= Budget
        # G = (Budget - P) / P
        # Let's pick P such that we have at least 5-10 generations.
        # If P=20, G=9. If P=10, G=19.
        # Larger P is generally better for diversity in multi-objective, but smaller P allows more generations.
        # For ZDT3, diversity is key. Let's try P=20.
        pop_size = 20
        if pop_size >= budget:
            pop_size = max(1, budget // 2)
            
        generations = (budget - pop_size) // pop_size
        
        # Initialize population
        rng = np.random.default_rng(0) # Use fixed seed for reproducibility if needed, but evaluator seeds np.random
        # The evaluator calls np.random.seed(seed). We should use np.random directly or default_rng?
        # The prompt says "evaluator calls np.random.seed(seed)". It's safer to use np.random functions
        # or default_rng seeded from the global state? 
        # Usually, it's best to use the global np.random state to respect the evaluator's seed.
        # However, default_rng is faster and more modern. But if the evaluator seeds np.random, 
        # default_rng(0) ignores that. 
        # Let's use np.random to be consistent with the seed setting.
        
        X = lo + range_ * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        
        used_evals = pop_size
        
        # NSGA-II Loop
        for gen in range(generations):
            # Create offspring
            Q = []
            for _ in range(pop_size):
                # Binary Tournament Selection
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                # We need to maintain ranks and crowding distances from previous generation
                # For simplicity in this implementation, we will recompute or keep them.
                # Let's keep them from the last selection step.
                
                # Actually, we need to store ranks and crowding distances for the current parent population.
                # Let's initialize them for the first generation or update them at the end of each generation.
                
                # To keep the code clean, I will structure the loop to maintain 
                # self.ranks and self.crowds for the current population.
                
                pass
            
            # This approach is getting messy with state. Let's refactor to a cleaner NSGA-II implementation.
            
            break # Stop this thought process and write the full clean code below.

    def _non_dominated_sort(self, F):
        """
        Performs non-dominated sorting on the population.
        Returns a list of fronts, where each front is a list of indices.
        Also returns the rank of each individual.
        """
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
        n_doms = np.zeros(n, dtype=int)  # n_doms[i] = number of individuals dominating i
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_doms[j] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_doms[i] += 1
        
        # Find first front (non-dominated solutions)
        current_front = [i for i in range(n) if n_doms[i] == 0]
        rank = np.full(n, -1, dtype=int)
        front_id = 0
        
        while current_front:
            fronts.append(current_front)
            for i in current_front:
                rank[i] = front_id
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_doms[j] -= 1
                    if n_doms[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_id += 1
            
        return fronts, rank

    def _crowding_distance(self, F, indices):
        """
        Calculates crowding distance for a set of indices in a front.
        Returns a dictionary of index -> distance.
        """
        if len(indices) <= 2:
            return {i: np.inf for i in indices}
            
        n_obj = self.n_obj
        dist = np.zeros(len(indices), dtype=float)
        
        for m in range(n_obj):
            # Sort by m-th objective
            sorted_indices = sorted(indices, key=lambda i: F[i][m])
            dist[indices.index(sorted_indices[0])] = np.inf
            dist[indices.index(sorted_indices[-1])] = np.inf
            
            f_min = F[sorted_indices[0]][m]
            f_max = F[sorted_indices[-1]][m]
            
            if f_max - f_min == 0:
                continue
                
            for k in range(1, len(sorted_indices) - 1):
                i = sorted_indices[k]
                # Find position of i in the sorted list to get neighbors
                # dist is indexed by position in the 'indices' list? No, let's use a dict or array aligned with indices.
                # Let's use an array aligned with the original population indices for simplicity in the main loop.
                # Actually, it's easier to return an array of distances for the whole population, 
                # setting 0 for those not in the front.
                
                # Let's change strategy: return an array of length N, where N is total population size.
                pass

        # Let's rewrite crowding distance to return an array of size N.
        return dist

    # The above helper functions are getting complicated to manage state.
    # Let's write a self-contained, efficient NSGA-II inside __call__.
    
    def _crowding_distance_array(self, F, front_indices, n):
        """
        Returns an array of crowding distances of size n.
        Only indices in front_indices are non-zero.
        """
        dist = np.zeros(n, dtype=float)
        if len(front_indices) <= 2:
            for i in front_indices:
                dist[i] = np.inf
            return dist
            
        n_obj = self.n_obj
        # Map original indices to positions in the front list for easy neighbor access?
        # No, let's just sort the front_indices by objective value.
        
        for m in range(n_obj):
            # Sort front indices by m-th objective
            sorted_front = sorted(front_indices, key=lambda idx: F[idx][m])
            
            # Set boundaries to inf
            dist[sorted_front[0]] = np.inf
            dist[sorted_front[-1]] = np.inf
            
            f_min = F[sorted_front[0]][m]
            f_max = F[sorted_front[-1]][m]
            
            if f_max - f_min == 0:
                continue
                
            for k in range(1, len(sorted_front) - 1):
                idx = sorted_front[k]
                prev_idx = sorted_front[k-1]
                next_idx = sorted_front[k+1]
                
                # Accumulate distance
                dist[idx] += (F[next_idx][m] - F[prev_idx][m]) / (f_max - f_min)
                
        return dist

    def _sbx_crossover(self, x1, x2, lo, hi):
        """
        Simulated Binary Crossover.
        Returns two children.
        """
        eta_c = 20.0
        u = np.random.rand(self.dim)
        v = np.random.rand(self.dim)
        
        c1 = np.zeros_like(x1)
        c2 = np.zeros_like(x2)
        
        for i in range(self.dim):
            if u[i] <= 0.5:
                alpha = (2 * u[i]) ** (1 / (eta_c + 1))
            else:
                alpha = (2 * (1 - u[i])) ** (1 / (eta_c + 1))
                
            beta = 1 + alpha * (x1[i] - x2[i]) / (hi[i] - lo[i]) if hi[i] != lo[i] else 1
            
            if abs(beta) > 1:
                beta_q = (2 - beta) ** (1 / (eta_c + 1)) if beta > 1 else (-2 * beta) ** (1 / (eta_c + 1))
            else:
                beta_q = beta ** (1 / (eta_c + 1))
                
            c1[i] = 0.5 * ((1 + beta_q) * x1[i] + (1 - beta_q) * x2[i])
            c2[i] = 0.5 * ((1 - beta_q) * x1[i] + (1 + beta_q) * x2[i])
            
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        """
        Polynomial Mutation.
        Returns mutated child.
        """
        eta_m = 20.0
        x_mut = x.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5: # Probability of mutation per variable is 1/dim usually, but let's use 1.0 for simplicity or 1/dim?
                # Standard NSGA-II uses 1/dim for each variable to mutate.
                # Let's use 1.0 probability for the whole vector? No, usually per variable.
                # Let's stick to per variable probability = 1.0 for simplicity in this short budget? 
                # Actually, 1/dim is standard.
                pass
        
        # Let's implement standard per-variable mutation
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i]) if hi[i] != lo[i] else 0
                delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i]) if hi[i] != lo[i] else 0
                
                u = np.random.rand()
                if u < 0.5:
                    xy = 1 + delta1
                    mut = 2 * u
                    delta = (xy ** (1 + 1/eta_m) - 1)
                else:
                    xy = 1 + delta2
                    mut = 2 * (1 - u)
                    delta = 1 - (xy ** (1 + 1/eta_m))
                    
                x_mut[i] += delta * (hi[i] - lo[i])
                
        x_mut = np.clip(x_mut, lo, hi)
        return x_mut

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 20
        if pop_size >= budget:
            pop_size = max(1, budget // 2)
            
        generations = (budget - pop_size) // pop_size
        
        # Initialize
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used_evals = pop_size
        
        # Main Loop
        for gen in range(generations):
            # Create offspring Q
            Q_X = []
            Q_F = []
            
            # Selection and Reproduction
            # We need ranks and crowding distances for the current population X
            fronts, ranks = self._non_dominated_sort(F)
            crowds = np.zeros(pop_size, dtype=float)
            for front in fronts:
                # Calculate crowding distance for this front
                # Note: _crowding_distance_array modifies an array of size N
                temp_dist = self._crowding_distance_array(F, front, pop_size)
                # Add to total crowds? No, crowding distance is only defined within a front.
                # So we just assign it.
                for idx in front:
                    crowds[idx] = temp_dist[idx]
                    
            # Binary Tournament Selection
            for _ in range(pop_size // 2):
                # Select two parents
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                
                # Compare ranks, then crowding
                if ranks[i1] < ranks[i2] or (ranks[i1] == ranks[i2] and crowds[i1] > crowds[i2]):
                    p1 = X[i1]
                else:
                    p1 = X[i2]
                    
                i3 = np.random.randint(pop_size)
                i4 = np.random.randint(pop_size)
                
                if ranks[i3] < ranks[i4] or (ranks[i3] == ranks[i4] and crowds[i3] > crowds[i4]):
                    p2 = X[i3]
                else:
                    p2 = X[i4]
                    
                # Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                # Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                
                Q_X.append(c1)
                Q_X.append(c2)
                Q_F.append(f1)
                Q_F.append(f2)
                
                used_evals += 2
                if used_evals >= budget:
                    break
                    
            if used_evals >= budget:
                break
            
            Q_X = np.array(Q_X)
            Q_F = np.array(Q_F)
            
            # Combine P and Q
            R_X = np.vstack((X, Q_X))
            R_F = np.vstack((F, Q_F))
            
            # Non-dominated sorting on R
            R_fronts, R_ranks = self._non_dominated_sort(R_F)
            
            # Select next generation
            next_X = []
            next_F = []
            n_selected = 0
            
            for front in R_fronts:
                if n_selected + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        next_X.append(R_X[idx])
                        next_F.append(R_F[idx])
                    n_selected += len(front)
                else:
                    # Take only some, based on crowding distance
                    # Calculate crowding distance for this front in R
                    temp_dist = self._crowding_distance_array(R_F, front, len(R_X))
                    
                    # Sort front indices by crowding distance (descending)
                    front_with_dist = [(idx, temp_dist[idx]) for idx in front]
                    front_with_dist.sort(key=lambda x: x[1], reverse=True)
                    
                    needed = pop_size - n_selected
                    for idx, _ in front_with_dist[:needed]:
                        next_X.append(R_X[idx])
                        next_F.append(R_F[idx])
                    n_selected += needed
                    break
                    
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
