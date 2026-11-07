"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using Differential Evolution for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_vars = self.dim
        n_obj = self.n_obj
        
        # Set population size
        # DE typically works well with pop_size between 10 and 50 for small budgets
        # We have 200 evaluations. Let's use pop_size = 20.
        # Generations = 200 / 20 = 10.
        # This allows for global exploration via DE mutation.
        pop_size = 20
        remaining_budget = self.budget

        # Initialize population uniformly
        X = lo + (hi - lo) * rng.random((pop_size, n_vars))
        F = np.array([problem(x) for x in X])
        remaining_budget -= pop_size

        # Parameters for DE
        CR = 0.9  # Crossover rate
        F_factor = 0.8  # Scaling factor

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            front = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                
                if n_count[i] == 0:
                    front[0].append(i)
            
            k = 0
            while front[k]:
                next_front = []
                for i in front[k]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    front.append(next_front)
            
            return front

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for individuals in a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = [0.0] * n
            for m in range(n_obj):
                obj_vals = np.array([F[i][m] for i in front_indices])
                sorted_idx = np.argsort(obj_vals)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range > 0:
                    for k in range(1, n - 1):
                        idx = sorted_idx[k]
                        dist[idx] += (obj_vals[sorted_idx[k+1]] - obj_vals[sorted_idx[k-1]]) / obj_range
            
            return dist

        # Main DE loop
        while remaining_budget >= pop_size:
            offspring_X = []
            offspring_F = []
            
            for i in range(pop_size):
                if remaining_budget < 1:
                    break
                
                # Select three distinct individuals different from i
                idxs = list(range(pop_size))
                idxs.remove(i)
                a, b, c = rng.choice(idxs, 3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = X[a] + F_factor * (X[b] - X[c])
                
                # Crossover: Binary crossover
                # Jrand is a random integer in [0, n_vars-1]
                Jrand = rng.integers(0, n_vars)
                trial = np.copy(X[i])
                for j in range(n_vars):
                    if rng.random() <= CR or j == Jrand:
                        trial[j] = v[j]
                
                # Bounds handling
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                f_trial = problem(trial)
                remaining_budget -= 1
                
                offspring_X.append(trial)
                offspring_F.append(f_trial)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Environmental selection: Select best pop_size from combined
            # Use non-dominated sorting and crowding distance
            fronts = fast_non_dominated_sort(combined_F)
            selected_indices = []
            
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    need = pop_size - len(selected_indices)
                    dist = crowding_distance(combined_F, front)
                    sorted_by_dist = sorted(range(len(front)), key=lambda i: dist[i], reverse=True)
                    selected_indices.extend(front[i] for i in sorted_by_dist[:need])
                    break
            
            X = combined_X[selected_indices]
            F = combined_F[selected_indices]

        # If we have remaining budget (which shouldn't happen if pop_size divides budget, but just in case)
        # Or if we didn't use all budget, we can do a few more random perturbations or just stop.
        # With pop_size=20 and budget=200, we use exactly 200.
        
        return X
# EVOLVE-BLOCK-END
