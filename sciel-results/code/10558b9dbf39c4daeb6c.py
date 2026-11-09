"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: balance exploration and exploitation
        # 200 budget allows for ~10 generations with pop=20
        pop_size = 20
        
        # Initial population
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper functions
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [0] * n
            front = [0]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[p] += 1
            
            for p in range(n):
                if n_count[p] == 0:
                    rank[p] = 0
                    front[0] += 1
            
            i = 0
            while front[i] > 0:
                next_front = []
                for p in S:
                    pass # S is list of lists, iterate differently
                # Correct iteration
                current_front_indices = [k for k in range(n) if rank[k] == i]
                for p in current_front_indices:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                front.append(len(next_front))
                
            return rank

        def crowding_distance(F, rank, indices):
            # Calculate crowding distance for points in the same front
            # Simplified: just return 0 for now if complex, or basic distance
            # For performance, we can use a simpler metric if needed, but standard CD is fine for n=20
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = [0.0] * n
            for m in range(n_obj):
                vals = [F[idx][m] for idx in indices]
                min_val = min(vals)
                max_val = max(vals)
                if max_val - min_val < 1e-12:
                    continue
                # Sort indices by this objective
                sorted_idx = sorted(indices, key=lambda i: F[i][m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (F[sorted_idx[k+1]][m] - F[sorted_idx[k-1]][m]) / (max_val - min_val)
            return dist

        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Combine parents and offspring
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                # Tournament selection
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                
                # We need ranks to do proper tournament. 
                # To save time, we can approximate dominance or just pick random if ranks are expensive.
                # With pop=20, fast_non_dominated_sort is O(n^2) which is 400 ops. Very fast.
                # Let's calculate ranks once per generation for selection.
                pass
            
            # Recalculate ranks for selection
            ranks = fast_non_dominated_sort(F)
            
            for _ in range(pop_size):
                # Tournament selection based on rank
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)
                if ranks[p1_idx] < ranks[p2_idx]:
                    p1_idx = p1_idx
                else:
                    p1_idx = p2_idx
                
                p3_idx = np.random.randint(pop_size)
                p4_idx = np.random.randint(pop_size)
                if ranks[p3_idx] < ranks[p4_idx]:
                    p3_idx = p3_idx
                else:
                    p3_idx = p4_idx
                
                # Crossover (SBX or simple blend)
                # Use simple arithmetic crossover
                x1 = X[p1_idx]
                x2 = X[p3_idx]
                
                # Blend
                alpha = np.random.rand()
                child1 = alpha * x1 + (1 - alpha) * x2
                child2 = (1 - alpha) * x1 + alpha * x2
                
                # Mutation
                # Polynomial-like or Gaussian
                sigma = 0.1
                child1 = child1 + sigma * (hi - lo) * np.random.randn(dim)
                child2 = child2 + sigma * (hi - lo) * np.random.randn(dim)
                
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                
                f1 = problem(child1)
                f2 = problem(child2)
                
                offspring_X.extend([child1, child2])
                offspring_F.extend([f1, f2])
                used += 2
                
                if used >= budget:
                    break
            
            if used >= budget:
                break

            # Combine populations
            X_all = np.vstack([X, np.array(offspring_X)])
            F_all = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sort on combined population
            ranks_all = fast_non_dominated_sort(F_all)
            
            # Select next population
            X_next = []
            F_next = []
            i = 0
            while len(X_next) < pop_size:
                # Get all individuals in rank i
                front_indices = [k for k in range(len(F_all)) if ranks_all[k] == i]
                
                if len(X_next) + len(front_indices) <= pop_size:
                    # Add all
                    for idx in front_indices:
                        X_next.append(X_all[idx])
                        F_next.append(F_all[idx])
                else:
                    # Need to truncate
                    # Calculate crowding distance for this front
                    cd = crowding_distance(F_all, ranks_all, front_indices)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front_indices, key=lambda idx: cd[front_indices.index(idx)], reverse=True)
                    # Add remaining
                    for idx in sorted_front:
                        if len(X_next) < pop_size:
                            X_next.append(X_all[idx])
                            F_next.append(F_all[idx])
                        else:
                            break
                
                i += 1
                if i > len(ranks_all):
                    break
            
            X = np.array(X_next)
            F = np.array(F_next)

        return X

# EVOLVE-BLOCK-END
