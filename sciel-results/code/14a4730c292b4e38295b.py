# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 20
        # Keep track of all evaluated points to return (though return is ignored, 
        # the evaluator records the calls to problem())
        
        # Initialize population
        X = lo + span * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Helper: Non-dominated sorting
        def non_dominated_sort(F_arr):
            n = len(F_arr)
            fronts = []
            S = [[] for _ in range(n)]  # Set of points dominated by i
            n_d = np.zeros(n, dtype=int)  # Number of points dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        n_d[i] += 1
            
            # First front
            front_idx = 0
            for i in range(n):
                if n_d[i] == 0:
                    rank[i] = front_idx
            
            while front_idx < n:
                current_front = [i for i in range(n) if rank[i] == front_idx]
                if not current_front:
                    break
                fronts.append(current_front)
                next_front_indices = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = front_idx + 1
                            next_front_indices.append(j)
                front_idx += 1
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F_arr, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.ones(n) * np.inf
            cd = np.zeros(n)
            for m in range(n_obj):
                # Sort front by objective m
                sorted_indices = sorted(front_indices, key=lambda i: F_arr[i][m])
                cd[sorted_indices[0]] = np.inf
                cd[sorted_indices[-1]] = np.inf
                f_min = F_arr[sorted_indices[0]][m]
                f_max = F_arr[sorted_indices[-1]][m]
                if f_max - f_min < 1e-10:
                    continue
                for k in range(1, n - 1):
                    idx = sorted_indices[k]
                    cd[idx] += (F_arr[sorted_indices[k + 1]][m] - F_arr[sorted_indices[k - 1]][m]) / (f_max - f_min)
            return cd

        # Main loop
        while used < budget:
            # Time check
            if time.time() > deadline:
                break

            # Selection: Tournament on (front, crowding)
            # Combine current population
            fronts = non_dominated_sort(F)
            # Assign crowding distances
            cd = np.zeros(len(F))
            for front in fronts:
                cd_front = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cd[idx] = cd_front[i]

            # Create offspring
            offspring_X = np.zeros((N, dim))
            offspring_F = np.zeros((N, n_obj))
            
            for i in range(N):
                if used >= budget or time.time() > deadline:
                    break
                
                # Tournament selection (size 2)
                p1_idx, p2_idx = rng.integers(0, N, size=2)
                # Select better: lower front rank, then higher crowding
                if rank[p1_idx] < rank[p2_idx] or (rank[p1_idx] == rank[p2_idx] and cd[p1_idx] > cd[p2_idx]):
                    p1, p2 = p1_idx, p2_idx
                else:
                    p1, p2 = p2_idx, p1_idx
                
                # Crossover (BLX-alpha)
                alpha = 0.5
                lo_blx = np.minimum(X[p1], X[p2]) - alpha * (np.maximum(X[p1], X[p2]) - np.minimum(X[p1], X[p2]))
                hi_blx = np.maximum(X[p1], X[p2]) + alpha * (np.maximum(X[p1], X[p2]) - np.minimum(X[p1], X[p2]))
                lo_blx = np.maximum(lo_blx, lo)
                hi_blx = np.minimum(hi_blx, hi)
                child = rng.uniform(lo_blx, hi_blx)
                
                # Mutation (Polynomial approximation with Gaussian)
                # Simple Gaussian mutation
                sigma = 0.1 * span
                child = child + sigma * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                offspring_X[i] = child
                f_val = problem(child)
                offspring_F[i] = f_val
                used += 1
            
            # Combine parents and offspring
            X_total = np.vstack([X, offspring_X])
            F_total = np.vstack([F, offspring_F])
            
            # Non-dominated sort total
            fronts_total = non_dominated_sort(F_total)
            
            # Select next population
            X_new = []
            F_new = []
            for front in fronts_total:
                if len(X_new) + len(front) <= N:
                    for idx in front:
                        X_new.append(X_total[idx])
                        F_new.append(F_total[idx])
                else:
                    # Need to select from this front based on crowding
                    cd_front = crowding_distance(F_total, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd_front[i], reverse=True)
                    needed = N - len(X_new)
                    for j in range(needed):
                        idx = sorted_front[j]
                        X_new.append(X_total[idx])
                        F_new.append(F_total[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)

        # Return final population (ignored by evaluator, but required)
        return X
# EVOLVE-BLOCK-END
