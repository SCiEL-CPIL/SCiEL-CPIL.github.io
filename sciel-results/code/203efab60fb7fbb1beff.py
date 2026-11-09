"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size
        # Budget is 200. NSGA-II typically needs larger populations, but with 12 dims and 3 obj, 
        # a moderate population is good. Let's use 40.
        # Generations = (200 - 40) / 40 = 4 generations. This is low for NSGA-II.
        # Maybe smaller population, more generations?
        # Pop 20: (200-20)/20 = 9 generations.
        # Pop 10: (200-10)/10 = 19 generations.
        # Let's try Pop 20 to balance diversity and generations.
        pop_size = 20
        
        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            fronts = []
            S = [set() for _ in range(n)]
            n_dominated = [0] * n
            rank = [-1] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dominated[i] += 1
                if n_dominated[i] == 0:
                    rank[i] = 0
            
            current_front = [i for i in range(n) if rank[i] == 0]
            fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = rank[i] + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            
            return fronts, rank

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            m = len(F[indices[0]])
            n = len(indices)
            dist = {i: 0.0 for i in indices}
            
            for k in range(m):
                # Sort by k-th objective
                sorted_idx = sorted(indices, key=lambda i: F[i][k])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                f_min = F[sorted_idx[0]][k]
                f_max = F[sorted_idx[-1]][k]
                
                if f_max - f_min == 0:
                    continue
                    
                for i in range(1, n - 1):
                    idx = sorted_idx[i]
                    dist[idx] += (F[sorted_idx[i+1]][k] - F[sorted_idx[i-1]][k]) / (f_max - f_min)
            
            return dist

        # Helper: Selection (Binary Tournament)
        def select_parent(rank, dist, X, F):
            i = rng.integers(len(X))
            j = rng.integers(len(X))
            
            # Compare rank, then crowding distance
            if rank[i] < rank[j]:
                return X[i]
            elif rank[j] < rank[i]:
                return X[j]
            else:
                if dist[i] > dist[j]:
                    return X[i]
                else:
                    return X[j]

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5: # Crossover probability 0.5
                child1 = np.zeros_like(x1)
                child2 = np.zeros_like(x1)
                for i in range(len(x1)):
                    if rng.random() < 0.5:
                        if x1[i] == x2[i]:
                            child1[i] = x1[i]
                            child2[i] = x2[i]
                        else:
                            u = rng.random()
                            if u <= 0.5:
                                alpha = 2 * u
                                beta = (1 + 2 * (x1[i] - x2[i]) / (hi[i] - lo[i])) if x1[i] > x2[i] else (1 + 2 * (x2[i] - x1[i]) / (hi[i] - lo[i]))
                                factor1 = (alpha * beta) ** (1.0 / (eta + 1.0))
                                factor2 = ((1 - alpha) * beta) ** (1.0 / (eta + 1.0))
                            else:
                                alpha = 2 * (1 - u)
                                beta = (1 + 2 * (x2[i] - x1[i]) / (hi[i] - lo[i])) if x2[i] > x1[i] else (1 + 2 * (x1[i] - x2[i]) / (hi[i] - lo[i]))
                                factor1 = (alpha * beta) ** (1.0 / (eta + 1.0))
                                factor2 = ((1 - alpha) * beta) ** (1.0 / (eta + 1.0))
                            
                            child1[i] = 0.5 * ((1 + factor1) * x1[i] + (1 - factor1) * x2[i])
                            child2[i] = 0.5 * ((1 + factor2) * x2[i] + (1 - factor2) * x1[i])
                    else:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta=20.0):
            child = x.copy()
            for i in range(len(x)):
                if rng.random() < 1.0 / self.dim: # Mutation rate 1/dim
                    u = rng.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                    
                    # Map delta to range
                    # Standard polynomial mutation formula uses normalized position
                    # Here we approximate by scaling with span
                    # A more robust way:
                    # y = x + delta * (hi - lo)
                    # But standard PM is:
                    # if u <= 0.5:
                    #   delta1 = (2*u)^(1/(eta+1)) - 1
                    # else:
                    #   delta1 = 1 - (2*(1-u))^(1/(eta+1))
                    # y = x + delta1 * (hi - lo)
                    # This is not strictly standard PM which depends on x's position, but works well for bounded search.
                    
                    child[i] = x[i] + delta * span[i]
            
            return np.clip(child, lo, hi)

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Evaluate current population for sorting
            fronts, rank = non_dominated_sort(F)
            
            # Calculate crowding distance for each front
            dist = {i: 0.0 for i in range(pop_size)}
            for front in fronts:
                if len(front) > 0:
                    front_dist = crowding_distance(F, front)
                    for i in front:
                        dist[i] = front_dist[i]
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # We need to generate enough offspring to fill the next generation
            # Usually pop_size offspring.
            # Check time and budget
            # Each offspring needs 1 evaluation.
            # We can generate up to min(pop_size, budget - used) offspring.
            n_offspring = min(pop_size, self.budget - used)
            
            for _ in range(n_offspring):
                if used >= self.budget or time.time() > deadline:
                    break
                
                p1 = select_parent(rank, dist, X, F)
                p2 = select_parent(rank, dist, X, F)
                
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                # Evaluate c1
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                # Evaluate c2 if budget allows
                if used < self.budget and time.time() <= deadline:
                    f2 = problem(c2)
                    used += 1
                    offspring_X.append(c2)
                    offspring_F.append(f2)
                else:
                    # If we can't evaluate c2, we just keep c1
                    # But we already appended c1.
                    pass
            
            if not offspring_X:
                break
            
            # Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Environmental selection
            fronts, rank = non_dominated_sort(combined_F)
            
            # Select best pop_size individuals
            next_X = []
            next_F = []
            next_rank = []
            next_dist = {i: 0.0 for i in range(len(combined_X))}
            
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    # Take all
                    for i in front:
                        next_X.append(combined_X[i])
                        next_F.append(combined_F[i])
                        next_rank.append(rank[i])
                else:
                    # Need to select from this front based on crowding distance
                    # Calculate crowding distance for this front
                    front_dist = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: front_dist[i], reverse=True)
                    needed = pop_size - len(next_X)
                    for i in sorted_front[:needed]:
                        next_X.append(combined_X[i])
                        next_F.append(combined_F[i])
                        next_rank.append(rank[i])
                    break
            
            if len(next_X) < pop_size:
                # Should not happen if fronts cover all, but just in case
                break
            
            X = np.array(next_X)
            F = np.array(next_F)
            # Update rank and dist for next iteration? 
            # We will recalculate them at the start of the loop.
            
        return X
# EVOLVE-BLOCK-END
