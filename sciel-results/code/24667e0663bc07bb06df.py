"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """NSGA-II Multi-objective Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        N = 20  # Population size
        # Check if budget allows at least one generation
        if self.budget < N:
            # If budget is very small, just do random search
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        # NSGA-II Parameters
        eta_c = 20.0
        eta_m = 20.0
        p_c = 0.9
        p_m = 1.0 / self.dim

        while used < self.budget:
            # Generate offspring
            offspring_X = np.zeros_like(X)
            offspring_F = np.zeros_like(F)
            
            # We generate N offspring
            for i in range(N):
                # Selection: Binary Tournament
                idx1, idx2 = self._tournament_select(F, rng)
                parent1 = X[idx1]
                parent2 = X[idx2]
                
                # Crossover
                if rng.random() < p_c:
                    child1, child2 = self._sbx_crossover(parent1, parent2, lo, hi, eta_c, rng)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._poly_mutate(child1, lo, hi, eta_m, p_m, rng)
                child2 = self._poly_mutate(child2, lo, hi, eta_m, p_m, rng)
                
                offspring_X[2*i] = child1
                offspring_X[2*i+1] = child2 if 2*i+1 < N else child1 # Handle odd N if needed, but N=20 is even
                
            # Since N is even, we have exactly N children from N/2 pairs? 
            # Wait, standard NSGA-II generates N offspring from N parents.
            # Usually, we do N/2 crossovers to get N children.
            # Let's adjust the loop to generate N children correctly.
            
            # Re-implement offspring generation to ensure exactly N children
            offspring_X = np.zeros((N, self.dim))
            for i in range(N // 2):
                idx1, idx2 = self._tournament_select(F, rng)
                parent1 = X[idx1]
                parent2 = X[idx2]
                
                if rng.random() < p_c:
                    child1, child2 = self._sbx_crossover(parent1, parent2, lo, hi, eta_c, rng)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self._poly_mutate(child1, lo, hi, eta_m, p_m, rng)
                child2 = self._poly_mutate(child2, lo, hi, eta_m, p_m, rng)
                
                offspring_X[2*i] = child1
                offspring_X[2*i+1] = child2

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += N
            
            # Combine parent and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Selection for next generation
            X, F = self._nsga2_select(X_combined, F_combined, N, rng)

        return X

    def _nsga2_select(self, X, F, N, rng):
        ranks, crowding_dist = self._fast_non_dominated_sort(F)
        
        # Sort by rank, then by crowding distance (descending)
        # We need to select N individuals
        # Create an array of indices sorted by (rank, -crowding_dist)
        indices = np.argsort(ranks - crowding_dist * 1e-6) # Hack: rank is integer, crowding is float. 
        # Better: sort by rank, then stable sort by -crowding within same rank?
        # Or just use lexsort.
        # We want low rank (better) and high crowding (better).
        # So sort by rank ascending, crowding descending.
        
        # np.lexsort sorts by the last key first.
        # So keys: (-crowding_dist, ranks)
        sorted_indices = np.lexsort((-crowding_dist, ranks))
        
        selected_indices = sorted_indices[:N]
        
        return X[selected_indices], F[selected_indices]

    def _tournament_select(self, F, rng):
        ranks, crowding_dist = self._fast_non_dominated_sort(F)
        N = len(F)
        idx1 = rng.integers(N)
        idx2 = rng.integers(N)
        
        if ranks[idx1] < ranks[idx2]:
            return idx1, idx2
        elif ranks[idx2] < ranks[idx1]:
            return idx2, idx1
        else:
            if crowding_dist[idx1] > crowding_dist[idx2]:
                return idx1, idx2
            else:
                return idx2, idx1

    def _fast_non_dominated_sort(self, F):
        N = len(F)
        n_obj = self.n_obj
        ranks = np.zeros(N, dtype=int)
        crowding_dist = np.zeros(N)
        
        S = [[] for _ in range(N)] # Dominated sets
        n = np.zeros(N, dtype=int) # Number of dominating solutions
        
        # Step 1: Dominance relations
        for i in range(N):
            for j in range(i + 1, N):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n[i] += 1
        
        # Step 2: Assign ranks
        front = []
        for i in range(N):
            if n[i] == 0:
                ranks[i] = 0
                front.append(i)
        
        k = 0
        while len(front) > 0:
            next_front = []
            for i in front:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            front = next_front
            k += 1
        
        # Step 3: Crowding distance
        for r in range(k + 1):
            front_indices = [i for i in range(N) if ranks[i] == r]
            if len(front_indices) <= 2:
                for i in front_indices:
                    crowding_dist[i] = np.inf
                continue
            
            for m in range(n_obj):
                # Sort front by objective m
                sorted_front = sorted(front_indices, key=lambda x: F[x][m])
                crowding_dist[sorted_front[0]] = np.inf
                crowding_dist[sorted_front[-1]] = np.inf
                
                f_min = F[sorted_front[0]][m]
                f_max = F[sorted_front[-1]][m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                    
                for i in range(1, len(sorted_front) - 1):
                    idx = sorted_front[i]
                    idx_prev = sorted_front[i - 1]
                    idx_next = sorted_front[i + 1]
                    crowding_dist[idx] += (F[idx_next][m] - F[idx_prev][m]) / f_range

        return ranks, crowding_dist

    def _sbx_crossover(self, p1, p2, lo, hi, eta, rng):
        child1 = p1.copy()
        child2 = p2.copy()
        
        for i in range(self.dim):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if rng.random() <= 1.0 / (eta + 1.0):
                        beta = (2.0 * rng.random()) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - rng.random()))) ** (1.0 / (eta + 1.0))
                    
                    child1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                    child2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
                    
                    # Clip
                    child1[i] = np.clip(child1[i], lo[i], hi[i])
                    child2[i] = np.clip(child2[i], lo[i], hi[i])
        return child1, child2

    def _poly_mutate(self, x, lo, hi, eta, p_m, rng):
        for i in range(self.dim):
            if rng.random() < p_m:
                delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                
                r = rng.random()
                if r <= 0.5:
                    xy = (2.0 * r + (1.0 - 2.0 * r) * (delta1 ** (eta + 1.0))) ** (1.0 / (eta + 1.0))
                    deltaq = (xy - 1.0) * (hi[i] - lo[i])
                else:
                    xy = (2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (delta2 ** (eta + 1.0))) ** (1.0 / (eta + 1.0))
                    deltaq = (xy - 1.0) * (hi[i] - lo[i])
                
                x[i] += deltaq
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x

# EVOLVE-BLOCK-END
