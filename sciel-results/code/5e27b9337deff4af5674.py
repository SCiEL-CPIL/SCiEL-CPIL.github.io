import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # With 200 budget, N=40 allows 5 generations (1 init + 4 offspring)
        # N=20 allows 10 generations.
        # In 12D, N=40 is likely better for diversity.
        N = 40
        
        if self.budget < 2 * N:
             # Fallback to random sampling if budget is too small for evolution
             X = lo + (hi - lo) * np.random.random((self.budget, self.dim))
             F = np.array([problem(x) for x in X])
             return X

        # Phase 1: LHS Initialization
        rng = np.random.default_rng(42)
        X = np.zeros((N, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(N)
            X[:, d] = (perm + 0.5) / N * (hi[d] - lo[d]) + lo[d]
        
        F = np.array([problem(x) for x in X])
        used = N

        # Phase 2: Evolutionary Loop
        while used + N <= self.budget:
            if time.time() > deadline:
                break
            
            offspring_X = np.empty_like(X)
            offspring_F = np.empty_like(F)
            
            for i in range(N):
                if time.time() > deadline:
                    break
                # Selection: Tournament of size 2
                idx1 = np.random.randint(0, N)
                idx2 = np.random.randint(0, N)
                if self._dominates(F[idx1], F[idx2]):
                    p1 = X[idx1]
                elif self._dominates(F[idx2], F[idx1]):
                    p1 = X[idx2]
                else:
                    p1 = X[np.random.choice([idx1, idx2])]
                    
                idx3 = np.random.randint(0, N)
                idx4 = np.random.randint(0, N)
                if self._dominates(F[idx3], F[idx4]):
                    p2 = X[idx3]
                elif self._dominates(F[idx4], F[idx3]):
                    p2 = X[idx4]
                else:
                    p2 = X[np.random.choice([idx3, idx4])]

                # Crossover: SBX
                if np.random.random() < 0.9:
                    child = self._sbx(p1, p2, lo, hi, 15.0)
                else:
                    child = p1.copy()
                    
                # Mutation: Polynomial
                child = self._poly_mutate(child, lo, hi, 20.0, 1.0/self.dim)
                
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1
                
                if used > self.budget:
                    break

            if used > self.budget:
                break

            # Combine and Select
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            X, F = self._nsga2_select(X_comb, F_comb, N)

        # Return the non-dominated points from the final population
        # Or just the final population, as the evaluator records all returned points?
        # "Every returned objective vector is recorded". 
        # If we return the whole population, dominated points are also recorded.
        # However, dominated points do not contribute to hypervolume if non-dominated ones are present.
        # It is safer to return only the non-dominated set to ensure we don't "pollute" the set, 
        # although the hypervolume calculation usually ignores dominated points.
        # But wait, the prompt says "Every returned objective vector is recorded".
        # If I return dominated points, they are in the list. The hypervolume calculation 
        # "Only points strictly better than the reference point in every objective count."
        # And typically hypervolume is calculated on the non-dominated subset.
        # To be safe and clean, I will return the non-dominated subset of the final population.
        
        nd_indices = self._non_dominated_indices(F)
        if len(nd_indices) == 0:
             return X
        return X[nd_indices]

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _sbx(self, p1, p2, lo, hi, eta):
        child = p1.copy()
        if np.random.random() > 0.5:
            p1, p2 = p2, p1
        for i in range(self.dim):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    x1, x2 = p1[i], p2[i]
                    if x1 > x2: x1, x2 = x2, x1
                    beta = 1.0
                    alpha = 2.0 - (x1 - lo[i]) / (x2 - x1 + 1e-14)
                    rand = np.random.random()
                    if rand <= 1.0 / (eta + 1.0):
                        beta = (2.0 * rand) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta + 1.0))
                    c1 = 0.5 * (x1 + x2 - beta * (x2 - x1))
                    c2 = 0.5 * (x1 + x2 + beta * (x2 - x1))
                    child[i] = np.clip(c1 if np.random.random() < 0.5 else c2, lo[i], hi[i])
        return child

    def _poly_mutate(self, x, lo, hi, eta, pm):
        for i in range(self.dim):
            if np.random.random() < pm:
                delta1 = (x[i] - lo[i]) / (hi[i] - lo[i] + 1e-14)
                delta2 = (hi[i] - x[i]) / (hi[i] - lo[i] + 1e-14)
                rand = np.random.random()
                if rand < 0.5:
                    xy = 1.0 - (1.0 - delta1) ** (eta + 1.0)
                else:
                    xy = 1.0 - (1.0 - delta2) ** (eta + 1.0)
                mut_pow = 1.0 / (eta + 1.0)
                xq = x[i] + (hi[i] - lo[i]) * (xy - 0.5)
                x[i] = np.clip(xq, lo[i], hi[i])
        return x

    def _non_dominated_indices(self, F):
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]: continue
            for j in range(n):
                if i == j: continue
                if self._dominates(F[j], F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def _nsga2_select(self, X, F, N):
        n = len(X)
        # Non-dominated sorting
        ranks = np.zeros(n, dtype=int)
        fronts = []
        remaining = set(range(n))
        
        while remaining:
            front = []
            for i in remaining:
                is_nd = True
                for j in remaining:
                    if i == j: continue
                    if self._dominates(F[j], F[i]):
                        is_nd = False
                        break
                if is_nd:
                    front.append(i)
            
            for i in front:
                ranks[i] = len(fronts)
                remaining.remove(i)
            
            fronts.append(front)
        
        # Crowding distance
        cd = np.zeros(n)
        for f in fronts:
            if len(f) <= 2:
                for i in f:
                    cd[i] = float('inf')
                continue
            for k in range(self.n_obj):
                sorted_indices = sorted(f, key=lambda i: F[i][k])
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                min_val = F[sorted_indices[0]][k]
                max_val = F[sorted_indices[-1]][k]
                range_val = max_val - min_val
                if range_val == 0: continue
                for j in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[j]
                    prev_idx = sorted_indices[j-1]
                    next_idx = sorted_indices[j+1]
                    cd[idx] += (F[next_idx][k] - F[prev_idx][k]) / range_val
        
        # Sort by rank, then crowding distance
        # Create list of (rank, -cd, index)
        sorted_indices = sorted(range(n), key=lambda i: (ranks[i], -cd[i]))
        
        # Take top N
        selected = sorted_indices[:N]
        return X[selected], F[selected]
