"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential-evolution multi-objective optimizer with non-dominated archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Larger initial population for better coverage
        pop_size = min(40, max(20, self.budget // 5))
        
        # Latin hypercube initialization
        X = np.zeros((pop_size, self.dim))
        for j in range(self.dim):
            perm = rng.permutation(pop_size)
            X[:, j] = lo[j] + span[j] * (perm + 0.5) / pop_size
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (a is better than b in at least one obj and no worse in others)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(point, F_set):
            """Check if point is dominated by any point in F_set. Return index of dominator or -1."""
            for i, f in enumerate(F_set):
                if dominates(f, point):
                    return i
            return -1

        def dominates_set(point, F_set):
            """Check if point dominates any point in F_set. Return list of indices."""
            idxs = []
            for i, f in enumerate(F_set):
                if dominates(point, f):
                    idxs.append(i)
            return idxs

        def crowding_distance(F_subset):
            """Compute crowding distance for a subset of points."""
            n = len(F_subset)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for k in range(self.n_obj):
                order = np.argsort(F_subset[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = F_subset[order[-1], k] - F_subset[order[0], k]
                if f_range == 0:
                    continue
                for m in range(1, n - 1):
                    cd[order[m]] += (F_subset[order[m + 1], k] - F_subset[order[m - 1], k]) / f_range
            return cd

        # Main DE loop
        while used < self.budget:
            # Identify non-dominated front for parent selection
            front_indices = []
            for i in range(pop_size):
                dominated_by = -1
                for j in range(pop_size):
                    if i != j and dominates(F[j], F[i]):
                        dominated_by = j
                        break
                if dominated_by == -1:
                    front_indices.append(i)
            
            parents = front_indices if len(front_indices) >= 3 else list(range(pop_size))
            n_par = len(parents)
            
            # Select parents
            i1 = parents[rng.integers(n_par)]
            i2 = parents[rng.integers(n_par)]
            i3 = parents[rng.integers(n_par)]
            base = parents[rng.integers(n_par)]
            
            # Ensure distinct indices if possible
            while i2 == i1:
                i2 = parents[rng.integers(n_par)]
            while i3 == i1 or i3 == i2:
                i3 = parents[rng.integers(n_par)]

            # Adaptive F parameter
            progress = used / self.budget
            F_param = 0.8 * (1.0 - 0.5 * progress) + 0.3 * progress

            # DE/rand/1 mutation
            mutant = X[base] + F_param * (X[i1] - X[i2] + X[i3] - X[base])
            # Small perturbation
            mutant += 0.01 * span * rng.standard_normal(self.dim)
            mutant = np.clip(mutant, lo, hi)

            f_mutant = problem(mutant)
            used += 1

            # Selection Strategy:
            # 1. If mutant dominates any points, replace the "worst" dominated point.
            # 2. Else if mutant is non-dominated, replace the point with lowest crowding distance in the non-dominated front.
            # 3. Else (mutant is dominated), discard.

            dominated_indices = dominates_set(f_mutant, F)
            
            if dominated_indices:
                # Replace the dominated point with the highest objective sum (worst)
                worst_idx = dominated_indices[0]
                worst_sum = np.sum(F[worst_idx])
                for idx in dominated_indices[1:]:
                    if np.sum(F[idx]) > worst_sum:
                        worst_sum = np.sum(F[idx])
                        worst_idx = idx
                X[worst_idx] = mutant
                F[worst_idx] = f_mutant
            else:
                # Check if mutant is dominated by existing points
                dom_by = is_dominated(f_mutant, F)
                if dom_by != -1:
                    # Mutant is worse, discard
                    continue
                
                # Mutant is non-dominated. Add to front and evict least crowded.
                # We need to find the non-dominated front in the *current* population
                # and include the mutant, then evict one.
                
                # For efficiency, just find the non-dominated front of current F
                current_front = []
                for i in range(pop_size):
                    is_dom = False
                    for j in range(pop_size):
                        if i != j and dominates(F[j], F[i]):
                            is_dom = True
                            break
                    if not is_dom:
                        current_front.append(i)
                
                # Combine current front with mutant
                # We need to find which point in the combined set has the lowest crowding distance
                # To do this, we simulate adding mutant and removing one.
                
                # Simple approach: Calculate crowding distance for current front.
                # If we add mutant, the distances change. 
                # Approximation: Find the point in current front that is "closest" to the mutant in objective space?
                # Or just pick the one with lowest CD in current front, and if mutant is better, keep mutant.
                
                # Better: Calculate CD for the set (Front + Mutant).
                # But we must remove one.
                
                # Let's just replace the point in the current front with the lowest crowding distance.
                # This is a standard heuristic.
                
                if len(current_front) == 0:
                    # Should not happen if pop is valid
                    replace_idx = 0
                else:
                    F_front = F[current_front]
                    cds = crowding_distance(F_front)
                    min_cd_idx_in_front = np.argmin(cds)
                    replace_idx = current_front[min_cd_idx_in_front]
                
                X[replace_idx] = mutant
                F[replace_idx] = f_mutant

        return X
# EVOLVE-BLOCK-END
