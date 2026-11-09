# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using NSGA-II with Differential Evolution mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_eval = 0

        # Helper to evaluate and count
        def eval_point(x):
            nonlocal n_eval
            x = np.clip(x, lo, hi)
            f = problem(x)
            n_eval += 1
            return x, f

        # Population size
        N = 20
        
        # Initialize population
        X = np.zeros((N, self.dim))
        F = np.zeros((N, self.n_obj))
        
        for i in range(N):
            x = lo + (hi - lo) * rng.random(self.dim)
            x, f = eval_point(x)
            X[i] = x
            F[i] = f
            
        # DE Parameters
        F_de = 0.8
        CR = 1.0 # We use binomial crossover with CR=1 effectively if we just replace, 
                 # but standard DE uses binomial. Let's use standard DE/1/bin.
        
        # NSGA-II Selection Helper
        def fast_non_dominated_sort(F_vals):
            """
            F_vals: (N, M) array of objective values.
            Returns: list of lists of indices, where each list is a front.
            """
            n = len(F_vals)
            # S[i] = set of solutions dominated by i
            S = [set() for _ in range(n)]
            # n[i] = number of solutions dominating i
            n_dom = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    # Check if i dominates j
                    # i dominates j if all f_i <= f_j and at least one <
                    if np.all(F_vals[i] <= F_vals[j]) and np.any(F_vals[i] < F_vals[j]):
                        S[i].add(j)
                        n_dom[j] += 1
                    elif np.all(F_vals[j] <= F_vals[i]) and np.any(F_vals[j] < F_vals[i]):
                        S[j].add(i)
                        n_dom[i] += 1
            
            # First front
            front = [i for i in range(n) if n_dom[i] == 0]
            if front:
                fronts.append(front)
            
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front
                
            return fronts

        def crowding_distance(F_vals, indices):
            """
            Calculate crowding distance for a set of indices.
            Returns a dict of index -> distance.
            """
            if not indices:
                return {}
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            m = self.n_obj
            distances = {i: 0.0 for i in indices}
            for k in range(m):
                # Sort indices by k-th objective
                sorted_idx = sorted(indices, key=lambda x: F_vals[x, k])
                # Assign infinite distance to boundaries
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                
                # Range of objective values
                f_range = F_vals[sorted_idx[-1], k] - F_vals[sorted_idx[0], k]
                if f_range == 0:
                    continue
                    
                for i in range(1, len(sorted_idx) - 1):
                    prev_idx = sorted_idx[i - 1]
                    next_idx = sorted_idx[i + 1]
                    distances[sorted_idx[i]] += (F_vals[next_idx, k] - F_vals[prev_idx, k]) / f_range
                    
            return distances

        # Main Loop
        # We have N=20, budget=200.
        # Init used 20. Remaining 180.
        # Each generation uses N=20 evaluations for offspring.
        # Generations = 180 // 20 = 9.
        
        while n_eval < self.budget:
            # Generate offspring
            X_off = np.zeros((N, self.dim))
            F_off = np.zeros((N, self.n_obj))
            
            for i in range(N):
                # DE/1/bin mutation
                # Select 3 distinct parents
                p1, p2, p3 = rng.choice(N, 3, replace=False)
                
                # v = X[p1] + F_de * (X[p2] - X[p3])
                v = X[p1] + F_de * (X[p2] - X[p3])
                
                # Binomial Crossover
                # For each dimension, with probability CR, take from v, else from X[i]
                # Here CR=0.5 is standard.
                cross_mask = rng.random(self.dim) < 0.5
                # Ensure at least one gene is taken from v
                if not np.any(cross_mask):
                    cross_mask[rng.integers(self.dim)] = True
                
                trial = np.where(cross_mask, v, X[i])
                trial = np.clip(trial, lo, hi)
                
                trial, f_trial = eval_point(trial)
                X_off[i] = trial
                F_off[i] = f_trial
                
                if n_eval >= self.budget:
                    break
            
            if n_eval >= self.budget:
                break
            
            # Combine Population and Offspring
            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            
            # Non-dominated Sort
            fronts = fast_non_dominated_sort(F_comb)
            
            # Select next population
            X_new = np.zeros((N, self.dim))
            F_new = np.zeros((N, self.n_obj))
            count = 0
            
            for front in fronts:
                if count + len(front) <= N:
                    # Take all
                    for idx in front:
                        X_new[count] = X_comb[idx]
                        F_new[count] = F_comb[idx]
                        count += 1
                else:
                    # Take subset based on crowding distance
                    cd = crowding_distance(F_comb, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda x: cd[x], reverse=True)
                    for idx in sorted_front:
                        if count < N:
                            X_new[count] = X_comb[idx]
                            F_new[count] = F_comb[idx]
                            count += 1
                        else:
                            break
                    break
                
                if count >= N:
                    break
            
            X = X_new
            F = F_new
            
        return X
# EVOLVE-BLOCK-END
