"""Rewritten program using Multi-Objective Differential Evolution (NSGA-II style)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II with Differential Evolution for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget
        
        # Parameters
        pop_size = 20
        cr = 0.5  # Crossover rate
        F = 0.8   # Differential weight
        
        # Initialize
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F_vals = np.array([problem(x) for x in X], dtype=float)
        used = pop_size
        
        # Helper: Fast Non-Dominated Sorting
        def fast_non_dominated_sort(Fs):
            m = len(Fs)
            S = [set() for _ in range(m)]
            n_count = [0] * m
            rank = [0] * m
            
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(Fs[i] <= Fs[j]) and np.any(Fs[i] < Fs[j]):
                        S[i].add(j)
                        n_count[i] += 1
                    elif np.all(Fs[j] <= Fs[i]) and np.any(Fs[j] < Fs[i]):
                        n_count[j] += 1
            
            front = [set() for _ in range(m)] # Temporary storage
            current_front = set()
            
            for i in range(m):
                if n_count[i] == 0:
                    rank[i] = 0
                    current_front.add(i)
            
            k = 0
            while current_front:
                front[k] = current_front
                next_front = set()
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.add(j)
                current_front = next_front
                k += 1
                
            # Convert to list of lists for easier handling
            fronts = [[] for _ in range(k)]
            for i in range(m):
                fronts[rank[i]].append(i)
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(front_indices, Fs):
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            m = len(front_indices)
            cd = {i: 0.0 for i in front_indices}
            n_obj = self.n_obj
            
            for d in range(n_obj):
                # Sort front indices by objective d
                sorted_indices = sorted(front_indices, key=lambda i: Fs[i][d])
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                
                obj_range = Fs[sorted_indices[-1]][d] - Fs[sorted_indices[0]][d]
                if obj_range > 1e-12:
                    for i in range(1, m - 1):
                        idx = sorted_indices[i]
                        cd[idx] += (Fs[sorted_indices[i+1]][d] - Fs[sorted_indices[i-1]][d]) / obj_range
                else:
                    for i in range(1, m - 1):
                        cd[sorted_indices[i]] = float('inf')
                        
            return cd

        # Main Loop
        while used < budget:
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # We can only create as many offspring as remaining budget allows
            # But NSGA-II typically creates N offspring.
            # If budget is not enough for full N, we create partial.
            # However, to maintain structure, let's create up to pop_size.
            # If used + pop_size > budget, we create (budget - used) offspring.
            
            n_offspring = min(pop_size, budget - used)
            
            for _ in range(n_offspring):
                # Select parents for DE mutation
                # Randomly select 3 distinct parents from current population
                i1, i2, i3 = rng.choice(pop_size, 3, replace=False)
                
                # DE/1/bin mutation
                v = X[i1] + F * (X[i2] - X[i3])
                
                # Binary Crossover
                mask = rng.random(n) < cr
                u = np.where(mask, v, X[i1])
                
                # Clip to bounds
                u = np.clip(u, lo, hi)
                
                # Evaluate
                f_u = problem(u)
                used += 1
                
                offspring_X.append(u)
                offspring_F.append(f_u)
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F_vals, offspring_F])
            
            # Non-dominated sorting on combined population
            fronts = fast_non_dominated_sort(combined_F)
            
            # Select next generation
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    # Add entire front
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(front, combined_F)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
                    
            X = np.array(new_X)
            F_vals = np.array(new_F)
            
        return X
# EVOLVE-BLOCK-END
