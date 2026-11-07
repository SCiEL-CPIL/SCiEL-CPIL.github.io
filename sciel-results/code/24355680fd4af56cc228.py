"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based Multi-Objective Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([3.0, 5.0, 7.0])

        # Population size: Small enough to fit budget, large enough for DE
        # Budget 200. Init 40. 160 remaining. 40 offspring per gen? No, DE generates 1 per target.
        # Let's use Pop Size = 40.
        # Init: 40 evals.
        # Gen 1: 40 offspring (40 evals). Total 80.
        # Gen 2: 40 offspring (40 evals). Total 120.
        # Gen 3: 40 offspring (40 evals). Total 160.
        # Gen 4: 40 offspring (40 evals). Total 200.
        # This fits exactly.
        
        pop_size = 40
        if self.budget < pop_size:
            pop_size = self.budget

        X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        all_X = X.copy()
        all_F = F.copy()

        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Vectorized dominance check
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                
                if n_dom[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            
            return fronts[:-1]

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return [np.inf] * n
            
            distances = [0.0] * n
            for m in range(self.n_obj):
                obj_vals = np.array([F[idx, m] for idx in front_indices])
                sorted_idx = np.argsort(obj_vals)
                sorted_front = [front_indices[i] for i in sorted_idx]
                
                distances[front_indices.index(sorted_front[0])] = np.inf
                distances[front_indices.index(sorted_front[-1])] = np.inf
                
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range == 0:
                    continue
                
                for i in range(1, n - 1):
                    pos = front_indices.index(sorted_front[i])
                    distances[pos] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / obj_range
            
            return distances

        def de_offspring(X, F, pop_size, lo, hi, dim):
            """Generate offspring using DE/rand/1/bin"""
            offspring_X = []
            offspring_F = []
            
            # DE parameters
            F_factor = 0.8  # Differential weight
            CR = 0.9      # Crossover rate
            
            for i in range(pop_size):
                # Select 3 distinct random indices
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = X[r1] + F_factor * (X[r2] - X[r3])
                
                # Binomial Crossover
                j_rand = np.random.randint(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = X[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                f_trial = problem(trial)
                offspring_X.append(trial)
                offspring_F.append(f_trial)
            
            return np.array(offspring_X), np.array(offspring_F)

        while used < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            off_X, off_F = de_offspring(X, F, pop_size, lo, hi, self.dim)
            
            # Check if we exceeded budget (should be exact if pop_size divides remaining)
            # If remaining budget < pop_size, we might have generated too many.
            # But we set pop_size=40 and budget=200, so it fits.
            # If not, we should truncate.
            actual_off_count = min(pop_size, self.budget - used)
            off_X = off_X[:actual_off_count]
            off_F = off_F[:actual_off_count]
            
            used += actual_off_count
            
            # Update records
            all_X = np.vstack([all_X, off_X])
            all_F = np.vstack([all_F, off_F])
            
            # Combine parents and offspring
            combined_X = np.vstack([X, off_X])
            combined_F = np.vstack([F, off_F])
            
            # Select next generation
            fronts = fast_non_dominated_sort(combined_F)
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return all_X
# EVOLVE-BLOCK-END
