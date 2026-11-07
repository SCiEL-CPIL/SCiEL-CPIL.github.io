"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Differential Evolution optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size tuned for budget 200, dim 10
        # N=10 allows ~18 generations (10 init + 18*10)
        N = 10
        if self.budget < N:
            N = max(1, self.budget)
            
        # DE Parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F_vals = np.array([problem(x) for x in X])
        used = N
        
        def dominates(a, b):
            """Returns True if a dominates b."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_d = np.zeros(n, dtype=int)
            front = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
                if n_d[i] == 0:
                    front.append(i)
            
            fronts = []
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                front = next_front
            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            n = len(indices)
            dist = np.zeros(n)
            for m in range(self.n_obj):
                sorted_idx = np.argsort([F[i, m] for i in indices])
                sorted_indices = [indices[i] for i in sorted_idx]
                
                min_val = F[sorted_indices[0], m]
                max_val = F[sorted_indices[-1], m]
                range_val = max_val - min_val
                if range_val == 0:
                    continue
                    
                dist[0] = np.inf
                dist[-1] = np.inf
                
                for i in range(1, n - 1):
                    dist[i] += (F[sorted_indices[i+1], m] - F[sorted_indices[i-1], m]) / range_val
            return dist

        while used < self.budget:
            # Generate offspring
            for i in range(N):
                if used >= self.budget:
                    break
                
                # Select two distinct random indices different from i
                r1 = np.random.randint(N)
                while r1 == i:
                    r1 = np.random.randint(N)
                r2 = np.random.randint(N)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(N)
                
                # DE/1/bin
                v = X[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < CR or j == np.random.randint(self.dim):
                        v[j] = X[i][j] + F * (X[r1][j] - X[r2][j])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Evaluate
                f_v = problem(v)
                used += 1
                
                # Replacement
                if dominates(f_v, F_vals[i]):
                    X[i] = v
                    F_vals[i] = f_v
                elif dominates(F_vals[i], f_v):
                    pass # Keep old
                else:
                    # Non-dominated
                    # Simple strategy: Keep the one with smaller sum of objectives
                    # Or use a small archive? 
                    # To maintain diversity, we can check distance to other non-dominated points in current pop.
                    # For simplicity and speed, we use a probabilistic choice or sum-based.
                    # ZDT6 front is convex. Sum of objectives is a good proxy for distance to ideal (0,0).
                    if np.sum(f_v) < np.sum(F_vals[i]):
                        X[i] = v
                        F_vals[i] = f_v
                    else:
                        # To ensure diversity, we might want to keep the one that is further from the centroid
                        # But for small budget, sum is often sufficient.
                        # Let's add a small random factor to escape local optima
                        if np.random.rand() < 0.1:
                            X[i] = v
                            F_vals[i] = f_v

        return X
# EVOLVE-BLOCK-END
