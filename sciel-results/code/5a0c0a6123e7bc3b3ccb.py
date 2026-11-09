"""Multi-objective optimizer on RE37 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-objective optimizer using NSGA-II with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """
        Fast Non-Dominated Sorting.
        Returns a list of fronts, where each front is a list of indices.
        """
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]  # S[i] contains indices dominated by i
        nd = np.zeros(n, dtype=int) # nd[i] is the number of solutions dominating i
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                # Check if p dominates q
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                # Check if q dominates p
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
        
        front = [i for i in range(n) if nd[i] == 0]
        if not front:
            # Fallback if no non-dominated points found (shouldn't happen)
            front = [0]
        
        fronts = [front]
        i = 0
        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front
            i += 1
        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for a set of indices in F.
        Returns an array of distances corresponding to the input indices.
        """
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            
            min_val = F_sub[order[0], m]
            max_val = F_sub[order[-1], m]
            if max_val == min_val:
                continue
                
            for k in range(1, n - 1):
                dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / (max_val - min_val)
        
        return dist

    def _sbx_crossover(self, x1, x2, bounds, eta=20.0):
        """
        Simulated Binary Crossover.
        Returns two children.
        """
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        x1 = np.clip(x1, lo, hi)
        x2 = np.clip(x2, lo, hi)
        
        # For each dimension, decide if crossover happens
        # Probability of crossover is 0.9 per dimension? Or 0.5 per pair?
        # Standard SBX: For each dimension, with prob 0.5, perform SBX.
        # Here we do it for all dimensions with a probability check.
        
        u = np.random.random(self.dim)
        # If u < 0.5, perform crossover for that dimension
        mask = u < 0.5
        
        # Initialize children as copies
        c1 = x1.copy()
        c2 = x2.copy()
        
        for i in range(self.dim):
            if not mask[i]:
                continue
                
            x1_i = x1[i]
            x2_i = x2[i]
            lo_i = lo[i]
            hi_i = hi[i]
            
            if x1_i == x2_i:
                continue
                
            # Calculate beta
            # beta = (2*u)^beta or (2*(1-u))^(-beta)
            u_i = np.random.random()
            
            if u_i <= 0.5:
                beta = (2 * u_i) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2 * (1.0 - u_i))) ** (1.0 / (eta + 1.0))
                
            # Standard SBX formulas
            # c1 = 0.5 * ((1+beta)*x1 + (1-beta)*x2)
            # c2 = 0.5 * ((1-beta)*x1 + (1+beta)*x2)
            
            c1[i] = 0.5 * ((1 + beta) * x1_i + (1 - beta) * x2_i)
            c2[i] = 0.5 * ((1 - beta) * x1_i + (1 + beta) * x2_i)
            
            # Clip to bounds
            c1[i] = np.clip(c1[i], lo_i, hi_i)
            c2[i] = np.clip(c2[i], lo_i, hi_i)
            
        return c1, c2

    def _polynomial_mutation(self, x, bounds, eta=20.0, prob=0.1):
        """
        Polynomial Mutation.
        Mutates each dimension with probability 'prob'.
        """
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        x = np.clip(x, lo, hi)
        
        u = np.random.random(self.dim)
        mask = u < prob
        
        for i in range(self.dim):
            if not mask[i]:
                continue
                
            x_i = x[i]
            lo_i = lo[i]
            hi_i = hi[i]
            
            if hi_i == lo_i:
                continue
                
            u_i = np.random.random()
            
            # Calculate delta
            # delta = (2*u)^(1/(eta+1)) - 1 or 1 - (2*(1-u))^(1/(eta+1))
            if u_i <= 0.5:
                delta = (2 * u_i) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                delta = 1.0 - (2 * (1.0 - u_i)) ** (1.0 / (eta + 1.0))
                
            # Normalize delta by range
            # delta is in [-1, 1] roughly. 
            # Standard formula: x' = x + delta * (hi - lo)
            # But usually delta is scaled such that it covers the range.
            # Let's use the standard polynomial mutation formula:
            # y = x + (hi - lo) * delta
            # where delta is calculated as above.
            
            x[i] = x_i + delta * (hi_i - lo_i)
            x[i] = np.clip(x[i], lo_i, hi_i)
            
        return x

    def _tournament_selection(self, F, ranks, crowding, k=2):
        """
        Binary Tournament Selection.
        Returns the index of the winner.
        """
        n = len(F)
        i1 = np.random.randint(n)
        i2 = np.random.randint(n)
        
        # Compare ranks (lower is better)
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            # Same rank, compare crowding distance (higher is better)
            if crowding[i1] > crowding[i2]:
                return i1
            else:
                return i2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Configuration
        # Budget 200.
        # Initial LHS: 50
        # Population Size N: 30
        # Generations: 5 (5 * 30 = 150)
        # Total: 200
        
        n_init = 50
        n_pop = 30
        n_gens = (self.budget - n_init) // n_pop
        
        # 1. Initialization: LHS
        indices = np.arange(n_init)
        permuted_indices = []
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            permuted_indices.append(perm)
        
        X = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            X[:, d] = lo[d] + (permuted_indices[d] + 0.5) * (hi[d] - lo[d]) / n_init
            
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Check deadline
        if time.time() > deadline:
            return F
        
        # Main NSGA-II Loop
        for gen in range(n_gens):
            if time.time() > deadline - 10:
                break
            
            # Calculate ranks and crowding for current population
            fronts = self._fast_non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for rank_idx, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank_idx
            
            # Calculate crowding distance for each front
            crowding = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    dists = self._crowding_distance(F, front)
                    for k, idx in enumerate(front):
                        crowding[idx] = dists[k]
            
            # Generate Offspring
            X_offspring = []
            F_offspring = []
            
            # We need to generate n_pop offspring
            for _ in range(n_pop // 2):
                if time.time() > deadline - 5:
                    break
                
                # Select parents
                p1_idx = self._tournament_selection(F, ranks, crowding)
                p2_idx = self._tournament_selection(F, ranks, crowding)
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # Crossover
                c1, c2 = self._sbx_crossover(x1, x2, bounds)
                
                # Mutation
                c1 = self._polynomial_mutation(c1, bounds)
                c2 = self._polynomial_mutation(c2, bounds)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                
                X_offspring.append(c1)
                F_offspring.append(f1)
                X_offspring.append(c2)
                F_offspring.append(f2)
                used += 2
                
                if used >= self.budget:
                    break
            
            if not X_offspring:
                break
                
            X_offspring = np.array(X_offspring)
            F_offspring = np.array(F_offspring)
            
            # Combine Parents and Offspring
            X_combined = np.vstack([X, X_offspring])
            F_combined = np.vstack([F, F_offspring])
            
            # Non-dominated sort on combined population
            fronts = self._fast_non_dominated_sort(F_combined)
            
            # Select best N individuals
            X_next = []
            F_next = []
            selected_indices = []
            
            for front in fronts:
                if len(X_next) + len(front) <= n_pop:
                    # Add all
                    for idx in front:
                        X_next.append(X_combined[idx])
                        F_next.append(F_combined[idx])
                        selected_indices.append(idx)
                else:
                    # Need to select only some from this front
                    # Calculate crowding distance for this front
                    dists = self._crowding_distance(F_combined, front)
                    # Sort by crowding distance (descending)
                    order = np.argsort(dists)[::-1]
                    needed = n_pop - len(X_next)
                    for k in range(needed):
                        idx_in_front = order[k]
                        global_idx = front[idx_in_front]
                        X_next.append(X_combined[global_idx])
                        F_next.append(F_combined[global_idx])
                        selected_indices.append(global_idx)
                    break
            
            if not X_next:
                break
                
            X = np.array(X_next)
            F = np.array(F_next)
            
            # Check if we have used all budget
            if used >= self.budget:
                break
        
        return F
# EVOLVE-BLOCK-END
