"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using an NSGA-II inspired approach.
    Uses non-dominated sorting, crowding distance, SBX crossover, and polynomial mutation.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size
        # We need to leave some room for initialization and potentially final refinement
        # Let's aim for a population size that divides the remaining budget nicely
        pop_size = 10
        if self.budget < 2 * pop_size:
            pop_size = max(2, self.budget // 2)
            
        # 1. Initialize Population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: Calculate Crowding Distance
        def crowding_distance(F, ranks):
            n = len(F)
            if n == 0:
                return np.array([])
            distances = np.zeros(n)
            for r in np.unique(ranks):
                indices = np.where(ranks == r)[0]
                if len(indices) <= 2:
                    distances[indices] = np.inf
                    continue
                for m in range(self.n_obj):
                    obj_vals = F[indices, m]
                    min_val = np.min(obj_vals)
                    max_val = np.max(obj_vals)
                    if max_val == min_val:
                        distances[indices] = np.inf
                        continue
                    sorted_indices = indices[np.argsort(obj_vals)]
                    distances[sorted_indices[0]] = np.inf
                    distances[sorted_indices[-1]] = np.inf
                    for i in range(1, len(sorted_indices) - 1):
                        prev_idx = sorted_indices[i-1]
                        next_idx = sorted_indices[i+1]
                        curr_idx = sorted_indices[i]
                        distances[curr_idx] += (F[next_idx, m] - F[prev_idx, m]) / (max_val - min_val)
            return distances

        # Helper: Non-dominated Sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            front = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            
            for i in range(n):
                if n_dom[i] == 0:
                    ranks[i] = 0
                    front.append(i)
            
            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            ranks[j] = k + 1
                            next_front.append(j)
                front = next_front
                k += 1
            return ranks

        # Helper: Binary Tournament Selection
        def tournament_select(ranks, distances, k=2):
            candidates = rng.integers(0, len(ranks), size=k)
            best_idx = candidates[0]
            for c in candidates[1:]:
                if ranks[c] < ranks[best_idx]:
                    best_idx = c
                elif ranks[c] == ranks[best_idx]:
                    if distances[c] > distances[best_idx]:
                        best_idx = c
            return best_idx

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(self.dim):
                if x1[i] == x2[i]:
                    continue
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                c1 = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                c2 = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                
                if c1 < lo[i]: c1 = lo[i]
                if c1 > hi[i]: c1 = hi[i]
                if c2 < lo[i]: c2 = lo[i]
                if c2 > hi[i]: c2 = hi[i]
                
                child1[i] = c1
                child2[i] = c2
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutate(x, eta=20.0):
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                    x[i] += delta * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main Loop
        while used + pop_size <= self.budget:
            # Calculate ranks and distances for current population
            ranks = non_dominated_sort(F)
            distances = crowding_distance(F, ranks)
            
            # Generate Offspring
            children_X = []
            for _ in range(pop_size):
                p1_idx = tournament_select(ranks, distances)
                p2_idx = tournament_select(ranks, distances)
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                children_X.append(c1)
                children_X.append(c2)
            
            # Evaluate Offspring
            children_X = np.array(children_X)
            # We only need pop_size children, but we generated 2*pop_size.
            # We will evaluate all 2*pop_size to have a larger pool for selection?
            # No, budget is tight. Let's just evaluate pop_size children.
            # Actually, standard NSGA-II generates N offspring from N parents.
            # Let's take the first pop_size children.
            children_X = children_X[:pop_size]
            
            children_F = np.array([problem(x) for x in children_X])
            used += pop_size
            
            # Combine Parents and Offspring
            X_combined = np.vstack([X, children_X])
            F_combined = np.vstack([F, children_F])
            
            # Select New Population
            ranks_comb = non_dominated_sort(F_combined)
            distances_comb = crowding_distance(F_combined, ranks_comb)
            
            # Sort by rank, then by distance (descending)
            sort_indices = np.lexsort((distances_comb, ranks_comb))
            # We want lowest rank first, highest distance first.
            # lexsort uses the last key as primary. So (distances, ranks) means ranks is primary?
            # No, lexsort(keys) sorts by keys[0], then keys[1], etc.
            # We want to sort by ranks ascending, then distances descending.
            # So we can negate distances or use a custom sort.
            # Let's just use argsort on a tuple of (rank, -distance)
            sort_indices = np.lexsort((-distances_comb, ranks_comb))
            
            X = X_combined[sort_indices[:pop_size]]
            F = F_combined[sort_indices[:pop_size]]

        # If we have remaining budget, do some local search on the best point?
        # Or just return.
        # Let's use remaining budget to refine the best non-dominated point using a simple hill climbing.
        if used < self.budget:
            # Find the best point (lowest rank, highest distance)
            ranks = non_dominated_sort(F)
            distances = crowding_distance(F, ranks)
            best_idx = np.lexsort((-distances, ranks))[0]
            best_x = X[best_idx].copy()
            best_f = F[best_idx].copy()
            
            # Simple local search
            step = 0.05
            for _ in range(self.budget - used):
                neighbor = best_x + step * (hi - lo) * rng.standard_normal(self.dim)
                neighbor = np.clip(neighbor, lo, hi)
                nf = problem(neighbor)
                # Check if neighbor dominates best_f
                if np.all(nf <= best_f) and np.any(nf < best_f):
                    best_x = neighbor
                    best_f = nf
                # Or if it's non-dominated and we want to add it? 
                # For hypervolume, adding a non-dominated point is good.
                # But we are limited in budget. Let's just try to improve the best point.
        
        return X
# EVOLVE-BLOCK-END
