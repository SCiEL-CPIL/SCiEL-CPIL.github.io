"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective NSGA-II for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # NSGA-II Parameters
        # With budget 200, we can afford a population size of 20-40.
        # Let's try pop_size = 20.
        pop_size = 20
        
        # Initialize Population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper functions
        def non_dominated_sort(F):
            """
            Perform non-dominated sorting on a set of objective vectors.
            Returns a list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [set() for _ in range(n)] # Dominated set
            n_dominates = [0] * n       # Number of solutions dominated by i
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dominates[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dominates[i] += 1
            
            current_front = [i for i in range(n) if n_dominates[i] == 0]
            fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
                current_front = next_front
            
            # If some points are not assigned (should not happen in minimization unless all dominated by each other? No, non-dominated sorting covers all)
            # Ensure all points are in some front
            assigned = set()
            for front in fronts:
                assigned.update(front)
            if len(assigned) < n:
                # This shouldn't happen, but just in case
                remaining = [i for i in range(n) if i not in assigned]
                fronts.append(remaining)
                
            return fronts

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for solutions in a front.
            Returns an array of distances corresponding to front_indices.
            """
            if len(front_indices) <= 2:
                return [float('inf')] * len(front_indices)
            
            distances = [0.0] * len(front_indices)
            n_obj = self.n_obj
            
            for m in range(n_obj):
                # Sort indices by objective m
                sorted_indices = sorted(front_indices, key=lambda i: F[i][m])
                
                # Boundary solutions get infinite distance
                distances[front_indices.index(sorted_indices[0])] = float('inf')
                distances[front_indices.index(sorted_indices[-1])] = float('inf')
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                
                if f_max - f_min == 0:
                    continue
                    
                for k in range(1, len(sorted_indices) - 1):
                    i = sorted_indices[k]
                    # Find previous and next in sorted order
                    prev_i = sorted_indices[k-1]
                    next_i = sorted_indices[k+1]
                    
                    dist = (F[next_i][m] - F[prev_i][m]) / (f_max - f_min)
                    distances[front_indices.index(i)] += dist
            
            return distances

        def tournament_selection(population_indices, F, fronts, crowding_dists):
            """
            Select one individual using binary tournament based on rank and crowding distance.
            """
            i = population_indices[rng.integers(0, len(population_indices))]
            j = population_indices[rng.integers(0, len(population_indices))]
            
            # Find rank (front index)
            rank_i = 0
            rank_j = 0
            for idx, front in enumerate(fronts):
                if i in front:
                    rank_i = idx
                    break
                if j in front:
                    rank_j = idx
                    break
            
            if rank_i < rank_j:
                return i
            elif rank_j < rank_i:
                return j
            else:
                # Same rank, use crowding distance
                # Need to map index to distance
                # crowding_dists is a dict or list? Let's make it a dict {index: distance}
                dist_i = crowding_dists.get(i, 0)
                dist_j = crowding_dists.get(j, 0)
                if dist_i > dist_j:
                    return i
                else:
                    return j

        def crossover(x1, x2):
            """Simulated Binary Crossover (SBX) approximation or simple blend."""
            # Simple Blend Crossover (BLX-alpha) or just uniform with mutation
            # Let's use a simple crossover: mix with probability
            # Or use NSGA-II standard SBX.
            # For simplicity and speed with dim=10, let's use a simple crossover
            # that creates two children.
            
            # Simulated Binary Crossover is complex. Let's use a simple arithmetic crossover
            # or just uniform crossover.
            # Given the small dim, uniform crossover is often fine if mutation is strong.
            
            r = rng.random(dim)
            child1 = np.where(r < 0.5, x1, x2)
            child2 = np.where(r < 0.5, x2, x1)
            
            return child1, child2

        def mutation(x):
            """Polynomial mutation or simple Gaussian mutation."""
            # Gaussian mutation
            # Standard deviation scaled by range
            std = 0.1 * (hi - lo)
            x_mut = x + rng.normal(0, std, dim)
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < budget:
            # Combine parent and offspring?
            # Standard NSGA-II:
            # 1. Create offspring of size N
            # 2. Combine P and Q to R (size 2N)
            # 3. Sort R into fronts
            # 4. Select N best from R to form new P
            
            # Since budget is tight, we can do a "generational" step.
            # Current population size: pop_size
            # We will generate pop_size offspring.
            # Total evaluations in this generation: pop_size
            # New population size: pop_size
            
            offspring = []
            offspring_F = []
            
            # Create offspring
            # We need to select parents from current population X, F
            # First, calculate fronts and crowding for current population to use in tournament
            current_indices = list(range(pop_size))
            fronts = non_dominated_sort(F)
            crowding_dists = {}
            for front in fronts:
                dists = crowding_distance(F, front)
                for idx, dist in zip(front, dists):
                    crowding_dists[idx] = dist
            
            for _ in range(pop_size):
                if used >= budget:
                    break
                p1 = tournament_selection(current_indices, F, fronts, crowding_dists)
                p2 = tournament_selection(current_indices, F, fronts, crowding_dists)
                
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutation(c1)
                c2 = mutation(c2)
                
                offspring.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                
                if used >= budget:
                    break
                offspring.append(c2)
                offspring_F.append(problem(c2))
                used += 1
            
            if not offspring:
                break
                
            # Combine
            all_X = np.vstack([X, np.array(offspring)])
            all_F = np.vstack([F, np.array(offspring_F)])
            
            # Sort all
            all_indices = list(range(len(all_X)))
            all_fronts = non_dominated_sort(all_F)
            
            # Select next population
            new_X = []
            new_F = []
            new_indices = []
            
            # Calculate crowding for all fronts to be used in selection
            # We need to pick pop_size individuals from all_fronts
            # Iterate through fronts
            selected_count = 0
            for front in all_fronts:
                if selected_count + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                        new_indices.append(idx)
                    selected_count += len(front)
                else:
                    # Need to select only some from this front
                    # Use crowding distance
                    dists = crowding_distance(all_F, front)
                    # Sort by distance descending
                    front_with_dists = list(zip(front, dists))
                    front_with_dists.sort(key=lambda x: x[1], reverse=True)
                    
                    needed = pop_size - selected_count
                    for i in range(needed):
                        idx = front_with_dists[i][0]
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                        new_indices.append(idx)
                    selected_count += needed
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
