"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, budget)
        
        # Initialize
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Returns list of non-dominated fronts."""
            n = len(F)
            S = [set() for _ in range(n)]  # Dominating sets
            n_dominates = [0] * n  # Number of solutions that dominate i
            fronts = [set()]  # First front

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dominates[j] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dominates[i] += 1

            for i in range(n):
                if n_dominates[i] == 0:
                    fronts[0].add(i)

            k = 0
            while len(fronts[k]) > 0:
                next_front = set()
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.add(j)
                k += 1
                fronts.append(next_front)

            # Remove last empty front
            if len(fronts[-1]) == 0:
                fronts.pop()

            return fronts

        def crowding_distance(front, F):
            """Returns crowding distance for solutions in front."""
            if len(front) <= 2:
                return np.full(len(front), np.inf)

            distances = np.zeros(len(front))
            front_indices = list(front)

            for m in range(self.n_obj):
                sorted_front = sorted(front_indices, key=lambda i: F[i][m])
                distances[sorted_front[0]] = np.inf
                distances[sorted_front[-1]] = np.inf

                f_min = F[sorted_front[0]][m]
                f_max = F[sorted_front[-1]][m]

                if f_max - f_min < 1e-10:
                    continue

                for k in range(1, len(sorted_front) - 1):
                    distances[sorted_front[k]] += (F[sorted_front[k+1]][m] - F[sorted_front[k-1]][m]) / (f_max - f_min)

            return distances

        def tournament_select(pop_X, pop_F, front_ids, crowding_dists, k=2):
            """Tournament selection based on rank and crowding distance."""
            idx1 = rng.integers(len(pop_X))
            idx2 = rng.integers(len(pop_X))
            
            # Compare by front rank first
            if front_ids[idx1] < front_ids[idx2]:
                return idx1
            elif front_ids[idx2] < front_ids[idx1]:
                return idx2
            else:
                # Same front, compare by crowding distance
                if crowding_dists[idx1] > crowding_dists[idx2]:
                    return idx1
                elif crowding_dists[idx2] > crowding_dists[idx1]:
                    return idx2
                else:
                    return idx1 if rng.random() < 0.5 else idx2

        while used < budget:
            if time.time() > deadline:
                break

            # Evaluate current population's fronts
            fronts = fast_non_dominated_sort(F)
            front_ids = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    front_ids[idx] = i

            # Calculate crowding distances for each front
            crowding_dists = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(front, F)
                    for i, idx in enumerate(front):
                        crowding_dists[idx] = cd[i]

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, budget - used)
            
            for _ in range(n_offspring):
                # Select parents
                p1 = tournament_select(X, F, front_ids, crowding_dists)
                p2 = tournament_select(X, F, front_ids, crowding_dists)
                
                # SBX crossover
                if rng.random() < 0.9:
                    child1 = np.zeros(n)
                    child2 = np.zeros(n)
                    for j in range(n):
                        if rng.random() < 0.5:
                            beta = 1.0
                            if F[p1][j] != F[p2][j]:
                                pass  # We don't have objective values per dimension, use uniform
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / 15)
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / 15)
                            child1[j] = 0.5 * ((1 + beta) * X[p1][j] + (1 - beta) * X[p2][j])
                            child2[j] = 0.5 * ((1 - beta) * X[p1][j] + (1 + beta) * X[p2][j])
                        else:
                            child1[j] = X[p1][j]
                            child2[j] = X[p2][j]
                    
                    # Mutation
                    for j in range(n):
                        if rng.random() < 1.0 / n:
                            sigma = (hi[j] - lo[j]) / 20
                            child1[j] += sigma * rng.standard_normal()
                            child2[j] += sigma * rng.standard_normal()
                    
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                    if used + 1 < budget:
                        offspring_F.append(problem(child1))
                        used += 1
                    if used + 1 < budget:
                        offspring_F.append(problem(child2))
                        used += 1
                    else:
                        offspring_F.pop()
                        offspring_X.pop()
                else:
                    # No crossover, just mutate
                    child = X[p1].copy()
                    for j in range(n):
                        if rng.random() < 1.0 / n:
                            sigma = (hi[j] - lo[j]) / 20
                            child[j] += sigma * rng.standard_normal()
                    child = np.clip(child, lo, hi)
                    offspring_X.append(child)
                    offspring_F.append(problem(child))
                    used += 1

            # Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])

            # Select next generation
            combined_fronts = fast_non_dominated_sort(combined_F)
            combined_front_ids = np.zeros(len(combined_F), dtype=int)
            for i, front in enumerate(combined_fronts):
                for idx in front:
                    combined_front_ids[idx] = i

            # Calculate crowding distances for combined population
            combined_crowding = np.zeros(len(combined_F))
            for front in combined_fronts:
                if len(front) > 0:
                    cd = crowding_distance(front, combined_F)
                    for i, idx in enumerate(front):
                        combined_crowding[idx] = cd[i]

            # Sort by front rank and crowding distance
            sorted_indices = sorted(range(len(combined_F)), key=lambda i: (combined_front_ids[i], -combined_crowding[i]))
            
            # Take first pop_size
            new_indices = sorted_indices[:pop_size]
            X = combined_X[new_indices]
            F = combined_F[new_indices]

        return X
# EVOLVE-BLOCK-END
