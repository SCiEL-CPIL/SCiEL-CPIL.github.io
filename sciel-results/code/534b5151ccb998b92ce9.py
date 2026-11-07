import numpy as np
import time

class Optimizer:
    """
    NSGA-II based multi-objective optimizer.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Fixed population size for NSGA-II
        # Must be less than or equal to budget
        self.pop_size = min(50, budget)
        
    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 325.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        pop_size = self.pop_size
        
        # 1. Initialization
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Helper functions for NSGA-II
        
        def fast_non_dominated_sort(F):
            """
            Returns ranks for each point in F.
            F: (N, M) array of objective values (minimization)
            """
            N = F.shape[0]
            M = F.shape[1]
            ranks = np.zeros(N, dtype=int)
            S = [[] for _ in range(N)]
            n_doms = np.zeros(N, dtype=int)
            front = []
            
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    # i dominates j if F[i] <= F[j] and F[i] != F[j]
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_doms[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_doms[j] += 1
                
                if n_doms[i] == 0:
                    front.append(i)
                    ranks[i] = 0
                    
            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            ranks[j] = k + 1
                            next_front.append(j)
                front = next_front
                k += 1
            return ranks

        def crowding_distance(F, ranks):
            """
            Calculate crowding distance for points in the same rank.
            """
            N = F.shape[0]
            M = F.shape[1]
            distances = np.zeros(N)
            
            unique_ranks = np.unique(ranks)
            for r in unique_ranks:
                indices = np.where(ranks == r)[0]
                n = len(indices)
                if n <= 2:
                    distances[indices] = np.inf
                    continue
                
                # Normalize objectives for this front
                F_front = F[indices]
                f_min = np.min(F_front, axis=0)
                f_max = np.max(F_front, axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1 # Avoid division by zero
                
                for i in range(n):
                    for m in range(M):
                        # Find neighbors in the sorted front for objective m
                        order = np.argsort(F_front[:, m])
                        pos = np.where(order == i)[0][0]
                        
                        if pos == 0 or pos == n - 1:
                            distances[indices[i]] += np.inf
                        else:
                            prev_idx = order[pos - 1]
                            next_idx = order[pos + 1]
                            distances[indices[i]] += (F_front[next_idx, m] - F_front[prev_idx, m]) / f_range[m]
            return distances

        def tournament_select(F, ranks, distances, k=2):
            """
            Select one individual via tournament.
            """
            idxs = np.random.randint(0, len(F), k)
            best_idx = idxs[0]
            for idx in idxs[1:]:
                # Compare (rank, -distance)
                if ranks[idx] < ranks[best_idx]:
                    best_idx = idx
                elif ranks[idx] == ranks[best_idx]:
                    if distances[idx] > distances[best_idx]:
                        best_idx = idx
            return best_idx

        def sbx_crossover(x1, x2, lo, hi, eta=15.0):
            """
            Simulated Binary Crossover.
            """
            x1, x2 = x1.copy(), x2.copy()
            if np.random.rand() < 0.5: # Crossover rate
                for i in range(len(x1)):
                    if np.random.rand() < 0.5: # Per-gene probability
                        if x1[i] != x2[i]:
                            if x1[i] > x2[i]:
                                x1[i], x2[i] = x2[i], x1[i]
                            
                            beta = 1.0
                            if np.random.rand() < 0.5:
                                beta = (2.0 * np.random.rand()) ** (1.0 / (eta + 1.0))
                            else:
                                beta = (1.0 / (2.0 * (1.0 - np.random.rand()))) ** (1.0 / (eta + 1.0))
                                
                            c1 = 0.5 * (1.0 + beta) * x1[i] + 0.5 * (1.0 - beta) * x2[i]
                            c2 = 0.5 * (1.0 - beta) * x1[i] + 0.5 * (1.0 + beta) * x2[i]
                            
                            c1 = np.clip(c1, lo[i], hi[i])
                            c2 = np.clip(c2, lo[i], hi[i])
                            
                            x1[i] = c1
                            x2[i] = c2
            return x1, x2

        def polynomial_mutation(x, lo, hi, eta=20.0):
            """
            Polynomial Mutation.
            """
            x = x.copy()
            for i in range(len(x)):
                if np.random.rand() < 1.0 / len(x):
                    if x[i] > lo[i] and x[i] < hi[i]:
                        if np.random.rand() < 0.5:
                            delta = (x[i] - lo[i]) / (hi[i] - lo[i])
                            delta_q = (1.0 + (eta + 1.0) * delta) ** (-1.0 / (eta + 1.0))
                            x[i] = x[i] - (hi[i] - lo[i]) * 0.5 * (1.0 - delta_q ** (eta + 1.0))
                        else:
                            delta = (hi[i] - x[i]) / (hi[i] - lo[i])
                            delta_q = (1.0 + (eta + 1.0) * delta) ** (-1.0 / (eta + 1.0))
                            x[i] = x[i] + (hi[i] - lo[i]) * 0.5 * (1.0 - delta_q ** (eta + 1.0))
            return np.clip(x, lo, hi)

        # Main Loop
        while evals < self.budget:
            # Check time
            if time.time() - start_time > timeout:
                break
            
            # Evaluate current population
            ranks = fast_non_dominated_sort(F)
            distances = crowding_distance(F, ranks)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # We generate pop_size offspring
            for _ in range(pop_size):
                # Selection
                p1_idx = tournament_select(F, ranks, distances)
                p2_idx = tournament_select(F, ranks, distances)
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx], lo, hi)
                
                # Mutation
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                # Evaluation
                f1 = problem(c1)
                f2 = problem(c2)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                
                evals += 2
                if evals >= self.budget:
                    break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Selection from combined population
            ranks_comb = fast_non_dominated_sort(F_combined)
            distances_comb = crowding_distance(F_combined, ranks_comb)
            
            # Sort by rank, then by distance (descending)
            # We need to select the best pop_size individuals
            indices = np.arange(len(F_combined))
            # Custom sort: primary key rank, secondary key -distance
            # Since we can't easily sort with numpy multi-key without struct, we use argsort twice or a stable sort
            # Sort by distance descending within same rank
            # Let's just use a simple stable sort approach
            # Actually, for small N, we can just iterate
            selected_indices = []
            current_rank = 0
            while len(selected_indices) < pop_size:
                # Get all individuals in current_rank
                front_indices = np.where(ranks_comb == current_rank)[0]
                
                if len(selected_indices) + len(front_indices) <= pop_size:
                    selected_indices.extend(front_indices.tolist())
                else:
                    # Need to select from this front based on crowding distance
                    # Sort front_indices by distance descending
                    dists = distances_comb[front_indices]
                    sorted_front = front_indices[np.argsort(-dists)]
                    needed = pop_size - len(selected_indices)
                    selected_indices.extend(sorted_front[:needed].tolist())
                    break
                current_rank += 1
            
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
            
        return X

# The task requires the class to be named Optimizer and have __init__ and __call__.
# The code above defines exactly that.
