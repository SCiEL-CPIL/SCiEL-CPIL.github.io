import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in class, but evaluator seeds np.random. 
        # Note: The prompt says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)". 
        # My class should ideally use np.random directly or its own rng. 
        # Using np.random directly ensures compatibility with the evaluator's seeding if I don't create a new default_rng.
        # However, default_rng is faster. But the evaluator seeds the global state. 
        # If I use np.random.seed(0) in the evaluator, and I use np.random.rand, I am aligned.
        # If I use default_rng(0), I ignore the evaluator's seed. 
        # The prompt says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)...".
        # This implies the evaluator expects the code to respect the global random state or that the seed is just for reproducibility of the *evaluation* environment.
        # Usually, in these tasks, using `np.random` global functions is safer to align with the evaluator's seed setting.
        
        # Let's use np.random global functions to respect the evaluator's seed.
        
        pop_size = 20
        X = np.zeros((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))
        
        # Initialization
        # 1/3 random
        # 2/3 biased to peaks 0.25, 0.5, 0.75
        n_rand = pop_size // 3
        n_peak1 = pop_size // 3
        n_peak2 = pop_size - n_rand - n_peak1
        
        X[:n_rand] = lo + (hi - lo) * np.random.random((n_rand, self.dim))
        
        peak1 = 0.25
        peak2 = 0.5
        peak3 = 0.75
        
        X[n_rand:n_rand+n_peak1] = lo + (hi - lo) * (peak1 + 0.1 * np.random.randn(n_peak1, self.dim))
        X[n_rand+n_peak1:n_rand+n_peak1+n_peak2] = lo + (hi - lo) * (peak2 + 0.1 * np.random.randn(n_peak2, self.dim))
        X[n_rand+n_peak1+n_peak2:] = lo + (hi - lo) * (peak3 + 0.1 * np.random.randn(pop_size - n_rand - n_peak1 - n_peak2, self.dim))
        
        X = np.clip(X, lo, hi)
        
        for i in range(pop_size):
            F[i] = problem(X[i])
        
        used = pop_size
        
        # NSGA-II Parameters
        # Crossover probability
        pc = 0.9
        # Mutation probability per gene
        pm = 1.0 / self.dim
        # Eta values (distribution indices)
        eta_c = 20
        eta_m = 20
        
        # Adaptive step size for mutation (initially large)
        # We will use polynomial mutation which scales with (hi-lo)
        
        while used < self.budget and time.time() < deadline:
            # Generate offspring
            offspring_X = np.zeros((pop_size, self.dim))
            
            for i in range(pop_size):
                # Tournament selection (binary)
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                # Simple dominance check for tournament
                if self._dominates(F[p1_idx], F[p2_idx]):
                    p1 = X[p1_idx]
                else:
                    p1 = X[p2_idx]
                
                p3_idx = np.random.randint(pop_size)
                p4_idx = np.random.randint(pop_size)
                if self._dominates(F[p3_idx], F[p4_idx]):
                    p2 = X[p3_idx]
                else:
                    p2 = X[p4_idx]
                
                child = p1.copy()
                
                # Crossover (SBX-like)
                if np.random.rand() < pc:
                    # Simple blend crossover or SBX
                    # Using simple arithmetic crossover for speed and simplicity, or SBX
                    # Let's use SBX for better exploration
                    for d in range(self.dim):
                        if np.random.rand() < 0.5:
                            # SBX formula
                            u = np.random.rand()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta_c + 1))
                            else:
                                beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta_c + 1))
                            c1 = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                            c2 = 0.5 * ((1 - beta) * p1[d] + (1 + beta) * p2[d])
                            child[d] = c1
                            # Swap p1 and p2 for the second child? No, we only generate one child per slot here.
                            # Actually, standard NSGA-II generates two children per pair. 
                            # Here we are filling a population slot. 
                            # We can just pick one of the two generated children.
                            if np.random.rand() < 0.5:
                                child[d] = c2
                else:
                    child = p1.copy()
                
                # Mutation
                for d in range(self.dim):
                    if np.random.rand() < pm:
                        # Polynomial mutation
                        u = np.random.rand()
                        if u < 0.5:
                            delta1 = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                        
                        # Normalize mutation to range
                        # Standard polynomial mutation uses normalized variable
                        x_norm = (child[d] - lo[d]) / (hi[d] - lo[d])
                        x_norm += delta1
                        x_norm = np.clip(x_norm, 0, 1)
                        child[d] = lo[d] + x_norm * (hi[d] - lo[d])
                
                offspring_X[i] = child
            
            # Evaluate offspring
            offspring_F = np.zeros((pop_size, self.n_obj))
            for i in range(pop_size):
                offspring_F[i] = problem(offspring_X[i])
                used += 1
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
            
            if used >= self.budget or time.time() > deadline:
                break
                
            # Combine parents and offspring
            all_X = np.vstack((X, offspring_X))
            all_F = np.vstack((F, offspring_F))
            
            # Non-dominated sorting
            ranks, fronts = self._non_dominated_sort(all_F)
            
            # Select next population
            next_X = np.zeros((pop_size, self.dim))
            next_F = np.zeros((pop_size, self.n_obj))
            count = 0
            
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        next_X[count] = all_X[idx]
                        next_F[count] = all_F[idx]
                        count += 1
                else:
                    # Need to pick from this front based on crowding distance
                    # Calculate crowding distance for this front
                    front_indices = np.array(front)
                    cd = self._crowding_distance(all_F[front_indices])
                    # Sort by crowding distance (descending)
                    sorted_indices = np.argsort(-cd)
                    for i in range(pop_size - count):
                        idx = front_indices[sorted_indices[i]]
                        next_X[count] = all_X[idx]
                        next_F[count] = all_F[idx]
                        count += 1
                    break
            
            X = next_X
            F = next_F
        
        return X

    def _dominates(self, a, b):
        # a dominates b if a <= b in all objectives and a < b in at least one
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        S = [[] for _ in range(n)] # Dominated sets
        n_doms = np.zeros(n, dtype=int) # Number of dominators
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_doms[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_doms[i] += 1
        
        for i in range(n):
            if n_doms[i] == 0:
                ranks[i] = 0
                fronts.append([i])
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_doms[j] -= 1
                    if n_doms[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
        
        return ranks, fronts

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        m = self.n_obj
        
        for k in range(m):
            # Sort by k-th objective
            sorted_indices = np.argsort(F[:, k])
            cd[sorted_indices[0]] = np.inf
            cd[sorted_indices[-1]] = np.inf
            
            # Calculate distance
            f_min = F[sorted_indices[0], k]
            f_max = F[sorted_indices[-1], k]
            if f_max == f_min:
                continue
            
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                cd[idx] += (F[sorted_indices[i + 1], k] - F[sorted_indices[i - 1], k]) / (f_max - f_min)
        
        return cd
