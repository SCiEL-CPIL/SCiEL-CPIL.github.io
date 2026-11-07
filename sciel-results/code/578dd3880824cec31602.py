"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II Multi-objective Optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        # Safety margin: finish well before 330s. 
        # The task says 300s limit for function, 330s kill. 
        # We use 300s as a hard stop to be safe.
        deadline = start_time + 290 

        # Determine population size
        # We want at least 5-10 generations.
        # Budget is 200. 
        # If N=20, generations = 10.
        # If N=10, generations = 20.
        # N=20 is a good balance for WFG12.
        N = 20
        
        # Check if budget allows for NSGA-II
        if budget < 2 * N:
            # Fallback to random search if budget is too small
            X = lo + (hi - lo) * rng.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Main loop
        while used + N <= budget:
            if time.time() > deadline:
                break
            
            # 1. Non-dominated sort and Crowding Distance for current population
            ranks, fronts = self._non_dominated_sort(F)
            cd = self._crowding_distance(F, fronts)
            
            # 2. Selection of parents via Tournament Selection
            # We need N offspring. 
            # Standard NSGA-II: Binary Tournament Selection for each parent.
            # We generate N/2 pairs of parents, then SBX crossover + mutation for 2 offspring each.
            parents = []
            for _ in range(N // 2):
                p1_idx = self._tournament_select(ranks, cd, rng)
                p2_idx = self._tournament_select(ranks, cd, rng)
                parents.append((p1_idx, p2_idx))
            
            # 3. Generate Offspring
            offspring_X = np.empty((N, dim))
            for i in range(N // 2):
                idx1, idx2 = parents[i]
                x1 = X[idx1]
                x2 = X[idx2]
                
                # SBX Crossover
                c1, c2 = self._sbx_crossover(x1, x2, lo, hi, rng)
                
                # Polynomial Mutation
                c1 = self._poly_mutation(c1, lo, hi, dim, rng)
                c2 = self._poly_mutation(c2, lo, hi, dim, rng)
                
                offspring_X[i*2] = c1
                offspring_X[i*2+1] = c2
            
            # 4. Evaluate Offspring
            F_off = np.array([problem(x) for x in offspring_X])
            used += N
            
            if time.time() > deadline:
                # If we just evaluated but are over deadline, we should still try to return 
                # the best we have. But the loop condition will catch it next time.
                # We can break here to avoid unnecessary computation if we are very close to limit.
                # However, we have the data. Let's just break the loop.
                break

            # 5. Combine and Select
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])
            
            ranks_all, fronts_all = self._non_dominated_sort(F_all)
            cd_all = self._crowding_distance(F_all, fronts_all)
            
            # Sort by rank (ascending) and crowding distance (descending)
            # We want the best N individuals.
            # Create a list of indices sorted by (rank, -cd)
            sort_keys = np.array([(ranks_all[i], -cd_all[i]) for i in range(len(F_all))])
            sort_idx = np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))
            
            # Select top N
            next_idx = sort_idx[:N]
            X = X_all[next_idx]
            F = F_all[next_idx]

        return X

    def _non_dominated_sort(self, F):
        """
        Fast Non-Dominated Sorting.
        Returns ranks (array of int) and fronts (list of lists of indices).
        """
        n = F.shape[0]
        ranks = np.zeros(n, dtype=int)
        fronts = []
        remaining = list(range(n))
        
        # Optimization: Precompute dominance relations? 
        # For N=40 (combined), N^2 = 1600 comparisons. 
        # Each comparison is O(M) where M=3. So ~5000 ops. Very fast.
        
        while remaining:
            front = []
            remaining_set = set(remaining)
            
            # Find non-dominated in remaining
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            
            if not front:
                # Fallback: assign all remaining to last front (should not happen in minimization with distinct points)
                front = remaining
                
            for i in front:
                ranks[i] = len(fronts)
            
            fronts.append(front)
            
            # Remove from remaining
            for i in front:
                remaining.remove(i)
                
        return ranks, fronts

    def _crowding_distance(self, F, fronts):
        """
        Calculates crowding distance for each point in F.
        """
        n = F.shape[0]
        n_obj = F.shape[1]
        cd = np.zeros(n)
        
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    cd[i] = np.inf
                continue
            
            # Get indices in this front
            idxs = np.array(front)
            F_front = F[idxs]
            
            min_vals = np.min(F_front, axis=0)
            max_vals = np.max(F_front, axis=0)
            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1.0 # Avoid division by zero
            
            # Normalize
            F_norm = (F_front - min_vals) / ranges
            
            # Initialize local CD for this front
            cd_local = np.zeros(len(front))
            
            for k in range(n_obj):
                # Sort indices in this front by k-th objective
                sorted_pos = np.argsort(F_norm[:, k])
                
                # First and last have infinite distance
                cd_local[sorted_pos[0]] = np.inf
                cd_local[sorted_pos[-1]] = np.inf
                
                if len(front) > 2:
                    for i in range(1, len(front) - 1):
                        # Distance between neighbors
                        diff = F_norm[sorted_pos[i+1], k] - F_norm[sorted_pos[i-1], k]
                        cd_local[sorted_pos[i]] += diff
            
            # Map back to global indices
            for i, global_idx in enumerate(idxs):
                cd[global_idx] = cd_local[i]
                
        return cd

    def _tournament_select(self, ranks, cd, rng):
        """
        Binary tournament selection.
        Returns index of the winner.
        """
        n = len(ranks)
        i1 = rng.integers(n)
        i2 = rng.integers(n)
        
        # Compare rank first
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            # Same rank, compare crowding distance (higher is better)
            if cd[i1] > cd[i2]:
                return i1
            else:
                return i2

    def _sbx_crossover(self, x1, x2, lo, hi, rng):
        """
        Simulated Binary Crossover (SBX).
        Returns two offspring.
        """
        eta = 20.0
        pc = 0.9
        dim = len(x1)
        
        c1 = np.zeros(dim)
        c2 = np.zeros(dim)
        
        for i in range(dim):
            if rng.random() < pc:
                if abs(x1[i] - x2[i]) < 1e-14:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    
                    # Clip to bounds
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
            else:
                c1[i] = x1[i]
                c2[i] = x2[i]
                
        return c1, c2

    def _poly_mutation(self, x, lo, hi, dim, rng):
        """
        Polynomial Mutation.
        """
        eta = 20.0
        pm = 1.0 / dim
        
        x_mut = x.copy()
        for i in range(dim):
            if rng.random() < pm:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1.0
                else:
                    delta = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                
                x_mut[i] = x[i] + delta * (hi[i] - lo[i])
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
                
        return x_mut
# EVOLVE-BLOCK-END
