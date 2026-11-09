"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Determine population size
        # With budget 200, N=50 allows for ~3 generations
        N = 50
        
        rng = np.random.default_rng(42)
        
        # 1. Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper functions for NSGA-II
        
        def fast_non_dominated_sort(F):
            """
            Assigns ranks to solutions.
            Returns list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_dom = np.zeros(n, dtype=int) # n_dom[i] = number of solutions dominated by i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[j] += 1
            
            fronts = []
            current_front = []
            for i in range(n):
                if n_dom[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = rank[i] + 1
                            next_front.append(j)
                current_front = next_front
            
            # Handle any unranked solutions (should not happen if logic is correct, but safe)
            for i in range(n):
                if rank[i] == 0 and i not in [idx for front in fronts for idx in front]:
                     # Fallback
                     pass 
            return fronts, rank

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices.
            Returns array of distances.
            """
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            m = len(indices)
            dist = np.zeros(m)
            for k in range(n_obj):
                # Sort indices by k-th objective
                obj_vals = F[indices, k]
                sorted_idx = np.argsort(obj_vals)
                sorted_obj = obj_vals[sorted_idx]
                
                if sorted_obj[0] == sorted_obj[-1]:
                    dist[sorted_idx] = np.inf
                    continue
                
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                for i in range(1, m - 1):
                    dist[sorted_idx[i]] += (sorted_obj[i+1] - sorted_obj[i-1]) / (sorted_obj[-1] - sorted_obj[0])
            
            return dist

        def select_parents(F, X, rank, dist):
            """
            Binary tournament selection.
            Returns two parent indices.
            """
            def tournament():
                i = rng.integers(N)
                j = rng.integers(N)
                # Compare rank first, then crowding distance
                if rank[i] < rank[j]:
                    return i
                elif rank[j] < rank[i]:
                    return j
                else:
                    if dist[i] > dist[j]:
                        return i
                    elif dist[j] > dist[i]:
                        return j
                    else:
                        return i if rng.random() < 0.5 else j
            
            p1 = tournament()
            p2 = tournament()
            return p1, p2

        def crossover(x1, x2):
            """
            Simulated Binary Crossover (SBX).
            """
            # eta_c = 20 is standard
            eta_c = 20
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            
            for i in range(dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        alpha = (2 * u) ** (1 / (eta_c + 1))
                    else:
                        alpha = (2 * (1 - u)) ** (1 / (eta_c + 1))
                    
                    betaq = 1 + (eta_c + 1) * (x1[i] - x2[i]) / (hi[i] - lo[i] + 1e-8)
                    betaq = max(1e-8, min(1e8, betaq))
                    beta = 1 / alpha * betaq if alpha * betaq <= 1 else 1 / (alpha * (2 - alpha * betaq))
                    
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
            
            return child1, child2

        def mutation(x):
            """
            Polynomial Mutation.
            """
            # eta_m = 20 is standard
            eta_m = 20
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < 0.9:  # Mutation rate 0.9
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    
                    x_mut[i] = x[i] + delta * (hi[i] - lo[i])
            
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < self.budget:
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Select parents and generate offspring
            # We need to generate N offspring to match population size
            # But we must respect budget.
            # Let's generate as many as possible up to N
            offspring_X = []
            offspring_F = []
            
            # We can generate N offspring if we have enough budget
            # But if budget is tight, we might generate fewer.
            # For simplicity and consistency with NSGA-II, let's try to generate N.
            # If budget < N, we break or generate partial.
            
            max_offspring = min(N, self.budget - used)
            
            for _ in range(max_offspring):
                if time.time() > deadline - 5:
                    break
                p1, p2 = select_parents(F, X, rank, dist)
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutation(c1)
                c2 = mutation(c2)
                
                f1 = problem(c1)
                f2 = problem(c2)
                used += 2
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used < self.budget:
                    offspring_X.append(c2)
                    offspring_F.append(f2)
                else:
                    break
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            fronts, rank_combined = fast_non_dominated_sort(F_combined)
            
            # Selection
            new_X = []
            new_F = []
            new_rank = np.zeros(len(F_combined), dtype=int)
            
            for i, front in enumerate(fronts):
                if len(new_X) + len(front) <= N:
                    # Add whole front
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                        new_rank[idx] = i
                else:
                    # Need to select from this front based on crowding distance
                    dist = crowding_distance(F_combined, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(range(len(front)), key=lambda k: dist[k], reverse=True)
                    needed = N - len(new_X)
                    for k in range(needed):
                        idx = front[sorted_front[k]]
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                        new_rank[idx] = i
                    break
            
            if not new_X:
                break
                
            X = np.array(new_X)
            F = np.array(new_F)
            
            # Recalculate rank and crowding distance for next iteration
            fronts, rank = fast_non_dominated_sort(F)
            # Crowding distance for the whole population? 
            # In NSGA-II, crowding is calculated per front during selection.
            # For selection in the next generation, we need rank and dist for all individuals.
            # Standard NSGA-II calculates dist for the fronts that are partially filled.
            # Here, we just need a metric for tournament.
            # We can calculate a global crowding distance approximation or just use rank.
            # Let's calculate dist for each front and store it.
            dist = np.zeros(N)
            for front in fronts:
                if len(front) > 1:
                    d = crowding_distance(F, front)
                    for k, idx in enumerate(front):
                        dist[idx] = d[k]
                else:
                    dist[front[0]] = np.inf
            
        return X
# EVOLVE-BLOCK-END
