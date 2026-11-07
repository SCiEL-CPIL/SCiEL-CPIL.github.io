"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using Differential Evolution (DE).
    DE is robust for deceptive landscapes and leverages population diversity.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Determine population size
        # With budget 200, we want several generations.
        # If N=50, we can do 4 generations (Init + 3 offspring sets).
        # If N=20, we can do 10 generations.
        # Smaller N allows more exploration steps but less diversity per step.
        # For WFG9, N=50 is a good balance.
        pop_size = 50
        if self.budget < pop_size:
            pop_size = self.budget
            
        # 1. Initialization: Stratified Sampling (Latin Hypercube)
        # This ensures coverage of the search space better than random.
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            # Create stratified samples for this dimension
            # Split [0, 1] into pop_size equal intervals
            samples = np.random.permutation(pop_size)
            X[:, d] = (samples + np.random.random(pop_size)) / pop_size
            X[:, d] = X[:, d] * range_[d] + lo[d]
            
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            current_front = [i for i in range(n) if n_dom[i] == 0]
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

        # Helper: Crowding Distance
        def crowding_distance(F, ranks, n):
            cd = np.zeros(n)
            unique_ranks = np.unique(ranks)
            for r in unique_ranks:
                idx = np.where(ranks == r)[0]
                if len(idx) <= 2:
                    cd[idx] = np.inf
                    continue
                for m in range(self.n_obj):
                    vals = F[idx, m]
                    min_val, max_val = np.min(vals), np.max(vals)
                    if max_val - min_val < 1e-10:
                        cd[idx] = np.inf
                        continue
                    order = np.argsort(vals)
                    sorted_idx = idx[order]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    for i in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[i]] += (vals[order[i+1]] - vals[order[i-1]]) / (max_val - min_val)
            return cd

        # DE Parameters
        F_factor = 0.8  # Differential weight
        CR = 0.9       # Crossover rate

        # Main Evolution Loop
        while used + pop_size <= self.budget and time.time() < deadline:
            # Generate offspring for each individual in population
            X_offspring = np.zeros_like(X)
            
            for i in range(pop_size):
                # Select 3 distinct random indices different from i
                r1, r2, r3 = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = X[r1] + F_factor * (X[r2] - X[r3])
                
                # Crossover: Binary Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.copy(X[i])
                for j in range(self.dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Bounds handling
                trial = np.clip(trial, lo, hi)
                X_offspring[i] = trial
            
            # Evaluate offspring
            F_offspring = np.array([problem(x) for x in X_offspring])
            used += pop_size
            
            # Selection: Compare offspring with parents
            # We want to keep the better of (parent, offspring)
            # "Better" is defined by non-domination and crowding distance
            # Simple strategy: If offspring dominates parent, replace.
            # If parent dominates offspring, keep parent.
            # If non-dominated, use crowding distance of the combined set?
            # For speed and simplicity, we can just check dominance.
            # To be more robust, we can sort the combined population (parents + offspring)
            # and take the best pop_size. But that requires sorting 2*N points.
            
            # Let's do the (mu+lambda) selection:
            # Combine X and X_offspring, F and F_offspring
            X_comb = np.vstack([X, X_offspring])
            F_comb = np.vstack([F, F_offspring])
            
            ranks = fast_non_dominated_sort(F_comb)
            cd = crowding_distance(F_comb, ranks, 2 * pop_size)
            
            # Sort indices by rank, then by crowding distance (descending)
            # We want to keep the top pop_size individuals
            # Create a list of indices sorted by (rank, -cd)
            indices = np.arange(2 * pop_size)
            # We need a sort key: (rank, -cd)
            # Since cd can be inf, we handle it.
            sort_keys = np.array([(ranks[i], -cd[i] if np.isfinite(cd[i]) else -1e10) for i in range(2 * pop_size)])
            
            # Sort indices based on keys
            # np.lexsort sorts by the last key first, so we reverse order or use tuple sort
            # Let's just use a simple argsort on a structured array or list
            sorted_indices = np.array(sorted(range(2 * pop_size), key=lambda i: (ranks[i], -cd[i])))
            
            # Take the first pop_size
            keep_indices = sorted_indices[:pop_size]
            
            X = X_comb[keep_indices]
            F = F_comb[keep_indices]

        # If we have remaining budget (e.g. if pop_size didn't divide budget evenly)
        # Or if we exited early due to time, we just return what we have.
        # The problem is that the evaluator records ALL calls.
        # So we must ensure we don't exceed budget.
        
        return X
# EVOLVE-BLOCK-END
