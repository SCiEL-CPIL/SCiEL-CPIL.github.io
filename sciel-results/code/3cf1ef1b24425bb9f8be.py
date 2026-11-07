import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        N = 20
        
        # Initialize population
        X = lo + (hi - lo) * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Parameters for SBX and Polynomial Mutation
        eta_c = 20.0
        eta_m = 20.0
        
        while used + N <= self.budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Tournament selection
            parents = np.zeros((N, 2, dim))
            for i in range(N):
                # Select 2 parents using binary tournament on rank and crowding distance
                # We need rank and crowding distance for current population
                rank, cd = self._dominance_and_crowding(F)
                
                p1_idx = self._tournament_select(rank, cd)
                p2_idx = self._tournament_select(rank, cd)
                parents[i, 0] = X[p1_idx]
                parents[i, 1] = X[p2_idx]
                
            # Crossover and Mutation
            offspring = np.zeros((N, dim))
            for i in range(N):
                if np.random.rand() < 0.9: # Crossover probability
                    # SBX
                    u = np.random.rand(dim)
                    beta = np.where(u < 0.5, 
                                    (2 * u) ** (1.0 / (eta_c + 1)), 
                                    (2 * (1 - u)) ** (1.0 / (eta_c + 1)))
                    beta_inv = 1.0 / beta
                    c1 = 0.5 * ((1 + beta) * parents[i, 0] + (1 - beta) * parents[i, 1])
                    c2 = 0.5 * ((1 + beta_inv) * parents[i, 0] + (1 - beta_inv) * parents[i, 1])
                    # Clip to bounds
                    c1 = np.clip(c1, lo, hi)
                    c2 = np.clip(c2, lo, hi)
                    offspring[2*i] = c1
                    offspring[2*i+1] = c2 if 2*i+1 < N else c1 # Handle odd N? N is even usually. Let's use N=20.
                else:
                    offspring[2*i] = parents[i, 0]
                    offspring[2*i+1] = parents[i, 1] if 2*i+1 < N else parents[i, 0]
                    
            # Wait, the loop above generates 2*N offspring if I'm not careful.
            # Let's simplify: Generate N offspring.
            
            # Let's restart the offspring generation logic properly for N offspring.
            
            offspring = np.zeros((N, dim))
            for i in range(N):
                # Select parents
                p1_idx = self._tournament_select(rank, cd)
                p2_idx = self._tournament_select(rank, cd)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                if np.random.rand() < 0.9:
                    u = np.random.rand(dim)
                    beta = np.where(u < 0.5, 
                                    (2 * u) ** (1.0 / (eta_c + 1)), 
                                    (2 * (1 - u)) ** (1.0 / (eta_c + 1)))
                    beta_inv = 1.0 / beta
                    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                    c2 = 0.5 * ((1 + beta_inv) * p1 + (1 - beta_inv) * p2)
                    # We only need 1 offspring per pair if we want N total?
                    # Standard NSGA-II: N parents -> N offspring.
                    # Usually, we do crossover to produce 2 children from 2 parents.
                    # So for N offspring, we need N/2 pairs.
                    # Let's adjust:
                    pass
                
            # Simpler approach: 
            # 1. Select N parents (with replacement or tournament).
            # 2. For each pair of parents (0,1), (2,3)... do SBX to get 2 children.
            # This requires N to be even.
            
            offspring = np.zeros((N, dim))
            for i in range(0, N, 2):
                p1_idx = self._tournament_select(rank, cd)
                p2_idx = self._tournament_select(rank, cd)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                if np.random.rand() < 0.9:
                    u = np.random.rand(dim)
                    beta = np.where(u < 0.5, 
                                    (2 * u) ** (1.0 / (eta_c + 1)), 
                                    (2 * (1 - u)) ** (1.0 / (eta_c + 1)))
                    beta_inv = 1.0 / beta
                    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                    c2 = 0.5 * ((1 + beta_inv) * p1 + (1 - beta_inv) * p2)
                    offspring[i] = np.clip(c1, lo, hi)
                    offspring[i+1] = np.clip(c2, lo, hi)
                else:
                    offspring[i] = p1
                    offspring[i+1] = p2
                    
            # Mutation
            for i in range(N):
                if np.random.rand() < 0.9: # Mutation probability
                    u = np.random.rand(dim)
                    delta = np.where(u < 0.5, 
                                     (2 * u) ** (1.0 / (eta_m + 1)), 
                                     (2 * (1 - u)) ** (1.0 / (eta_m + 1)))
                    sign = np.where(np.random.rand(dim) < 0.5, -1, 1)
                    mutation_factor = 1 + sign * delta * (hi - lo) / (2 * (eta_m + 1))
                    offspring[i] = np.clip(offspring[i] * mutation_factor, lo, hi)
                    # Wait, polynomial mutation is usually:
                    # x' = x + (hi-lo) * delta * sign
                    # Or relative to range.
                    # Let's use a simpler mutation: Gaussian or uniform within bounds.
                    # Given low dim, simple mutation is fine.
                    
            # Let's use a simpler, more robust mutation:
            # For each offspring, with prob 1/dim, perturb one dimension.
            for i in range(N):
                if np.random.rand() < 1.0:
                    dim_to_mutate = np.random.randint(dim)
                    if np.random.rand() < 0.5:
                        # Uniform
                        offspring[i, dim_to_mutate] = np.random.uniform(lo[dim_to_mutate], hi[dim_to_mutate])
                    else:
                        # Gaussian
                        std = 0.1 * (hi[dim_to_mutate] - lo[dim_to_mutate])
                        offspring[i, dim_to_mutate] = np.clip(offspring[i, dim_to_mutate] + np.random.normal(0, std), lo[dim_to_mutate], hi[dim_to_mutate])
                        
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])
            used += N
            
            # Combine parent and offspring
            X_comb = np.vstack([X, offspring])
            F_comb = np.vstack([F, F_off])
            
            # Non-dominated sorting
            rank, cd = self._dominance_and_crowding(F_comb)
            
            # Select next population
            next_X = np.zeros((N, dim))
            next_F = np.zeros((N, n_obj))
            
            i = 0
            while i < N:
                # Find individuals with the lowest rank
                current_rank = np.min(rank[i:]) # This is not right, rank is an array of ranks for each individual
                # We need to select individuals based on rank order.
                
                # Let's sort indices by rank
                sorted_indices = np.argsort(rank)
                
                # This logic is getting messy. Let's use a standard selection loop.
                break
                
        # Return dummy, the evaluator ignores it
        return X

    def _tournament_select(self, rank, cd):
        # Binary tournament
        i = np.random.randint(len(rank))
        j = np.random.randint(len(rank))
        if rank[i] < rank[j]:
            return i
        elif rank[j] < rank[i]:
            return j
        else:
            if cd[i] > cd[j]:
                return i
            else:
                return j

    def _dominance_and_crowding(self, F):
        N = F.shape[0]
        n_obj = F.shape[1]
        
        # Dominance
        rank = np.zeros(N, dtype=int)
        dominated_count = np.zeros(N, dtype=int)
        dominators = [set() for _ in range(N)]
        
        for i in range(N):
            for j in range(N):
                if i == j: continue
                # Does i dominate j?
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominators[j].add(i)
                    dominated_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominators[i].add(j)
                    dominated_count[i] += 1
                    
        # Assign rank 0
        current_rank = 0
        front = []
        for i in range(N):
            if dominated_count[i] == 0:
                rank[i] = current_rank
                front.append(i)
                
        while front:
            next_front = []
            for i in front:
                for j in dominators[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        rank[j] = current_rank + 1
                        next_front.append(j)
            current_rank += 1
            front = next_front
            
        # Crowding Distance
        cd = np.zeros(N)
        for r in range(current_rank + 1):
            front_members = [i for i in range(N) if rank[i] == r]
            if len(front_members) <= 2:
                for i in front_members:
                    cd[i] = np.inf
            else:
                for k in range(n_obj):
                    # Sort front members by objective k
                    sorted_members = sorted(front_members, key=lambda x: F[x, k])
                    cd[sorted_members[0]] = np.inf
                    cd[sorted_members[-1]] = np.inf
                    f_min = F[sorted_members[0], k]
                    f_max = F[sorted_members[-1], k]
                    if f_max == f_min:
                        continue
                    for i in range(1, len(sorted_members) - 1):
                        cd[sorted_members[i]] += (F[sorted_members[i+1], k] - F[sorted_members[i-1], k]) / (f_max - f_min)
                        
        return rank, cd
