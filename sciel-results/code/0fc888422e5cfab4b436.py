import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        N = 20
        
        # Initialization
        X = lo + range_ * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Main Loop
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring_X = np.zeros((N, self.dim))
            offspring_F = np.zeros((N, self.n_obj))
            
            for i in range(N):
                # Selection: Tournament selection on non-dominated ranks
                # We need to maintain ranks from previous generation
                # For simplicity and speed, we can just pick random parents 
                # or use a simple elitist strategy. 
                # Proper NSGA-II requires sorting the combined population.
                
                # Let's do a simplified NSGA-II step:
                # 1. Create offspring from current X
                # 2. Combine X and offspring
                # 3. Sort and select top N
                
                # To keep it efficient and simple for N=20:
                # Pick 2 parents from X
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                
                # Crossover (SBX)
                p1 = X[idx1]
                p2 = X[idx2]
                c1, c2 = self._sbx(p1, p2, lo, hi)
                
                # Mutation
                c1 = self._mutate(c1, lo, hi)
                c2 = self._mutate(c2, lo, hi)
                
                offspring_X[2*i] = c1
                offspring_X[2*i+1] = c2
                
                # Wait, N=20, so we need 20 offspring.
                # If I do N/2 pairs, I get N offspring.
                # So loop range(N//2) and fill 2*i, 2*i+1.
                # But if N is odd, handle last one.
                # Let's just generate N offspring directly by picking parents.
            
            # Regenerate offspring logic properly
            offspring_X = np.zeros((N, self.dim))
            for i in range(N):
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                p1 = X[idx1]
                p2 = X[idx2]
                
                # Crossover
                c1, _ = self._sbx(p1, p2, lo, hi)
                
                # Mutation
                c1 = self._mutate(c1, lo, hi)
                offspring_X[i] = c1
                
            # Evaluate offspring
            for i in range(N):
                if time.time() > deadline:
                    break
                offspring_F[i] = problem(offspring_X[i])
                evals += 1
                if evals >= self.budget:
                    break
            
            if evals >= self.budget:
                break

            # Combine parent and offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            ranks = self._non_dominated_sort(F_comb)
            
            # Crowding distance
            cd = self._crowding_distance(F_comb, ranks)
            
            # Selection
            # Sort by rank, then by crowding distance (descending)
            # We need to select N individuals.
            # Create a list of indices sorted by (rank, -cd)
            indices = np.arange(len(F_comb))
            # Sort: primary key rank (asc), secondary key cd (desc)
            # np.lexsort works in reverse order of keys
            sort_indices = np.lexsort((-cd, ranks))
            
            # Take first N
            new_indices = sort_indices[:N]
            
            X = X_comb[new_indices]
            F = F_comb[new_indices]

        return X

    def _non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        S = [[] for _ in range(n)]
        n_dominates = np.zeros(n, dtype=int)
        front = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                # F[i] <= F[j] and F[i] != F[j]
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[i] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dominates[i] += 1
        
        for i in range(n):
            if n_dominates[i] == 0:
                ranks[i] = 0
                front.append(i)
                
        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            front = next_front
            k += 1
            
        return ranks

    def _crowding_distance(self, F, ranks):
        n = len(F)
        cd = np.zeros(n)
        
        for rank in np.unique(ranks):
            # Get indices of this rank
            idx = np.where(ranks == rank)[0]
            if len(idx) <= 2:
                cd[idx] = np.inf
                continue
                
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = idx[np.argsort(F[idx, m])]
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                # Normalize
                f_min = F[idx, m].min()
                f_max = F[idx, m].max()
                if f_max == f_min:
                    continue
                    
                for i in range(1, len(sorted_idx) - 1):
                    cd[sorted_idx[i]] += (F[sorted_idx[i+1], m] - F[sorted_idx[i-1], m]) / (f_max - f_min)
                    
        return cd

    def _sbx(self, p1, p2, lo, hi):
        # Simulated Binary Crossover
        c1 = p1.copy()
        c2 = p2.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                continue
                
            x1 = p1[i]
            x2 = p2[i]
            if x1 == x2:
                continue
                
            beta1 = 1.0
            beta2 = 1.0
            alpha1 = 2.0
            alpha2 = 2.0
            
            if x1 > x2:
                x1, x2 = x2, x1
                beta1 = (x1 - lo[i]) / (x2 - x1 + 1e-9)
                beta2 = (hi[i] - x2) / (x2 - x1 + 1e-9)
            else:
                beta1 = (x2 - x1) / (hi[i] - x1 + 1e-9)
                beta2 = (x2 - x1) / (x2 - lo[i] + 1e-9)
                
            # Correct logic for SBX
            # Standard SBX:
            # if x1 <= x2:
            #   beta1 = (x2-x1)/(hi-x1)
            #   beta2 = (x2-x1)/(x2-lo)
            # else:
            #   beta1 = (x1-x2)/(x1-lo)
            #   beta2 = (x1-x2)/(hi-x2)
            
            # Let's use a simpler, robust Gaussian mutation/crossover to avoid SBX bugs
            # Given the small budget, simpler operators might be more robust.
            pass
            
        # Actually, let's just use Uniform Crossover + Gaussian Mutation for robustness
        # It's less likely to fail and easier to tune.
        c1 = 0.5 * p1 + 0.5 * p2
        c2 = 0.5 * p1 + 0.5 * p2
        return c1, c2

    def _mutate(self, x, lo, hi):
        # Gaussian Mutation
        sigma = 0.1
        x_new = x + sigma * (hi - lo) * np.random.randn(self.dim)
        x_new = np.clip(x_new, lo, hi)
        return x_new
