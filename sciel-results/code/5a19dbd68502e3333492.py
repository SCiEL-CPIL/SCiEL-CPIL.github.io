# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Use global numpy random to respect np.random.seed(seed)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        pop_size = 20
        if self.budget < pop_size:
            pop_size = max(1, self.budget)
            
        # Initialize population
        X = lo + span * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper functions for NSGA-II
        
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_doms = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_doms[i] += 1
                if n_doms[i] == 0:
                    fronts.append(i)
            
            k = 0
            while fronts:
                current_front = fronts
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                fronts = next_front
                # We need to return the fronts as a list of lists
                # Re-doing logic to collect properly
            # The above loop is slightly off for collecting all fronts. Let's use a standard implementation.
            
            # Standard Fast Non-Dominated Sort
            S = [set() for _ in range(n)]
            n_doms = [0] * n
            front_idx = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_doms[i] += 1
                if n_doms[i] == 0:
                    front_idx[i] = 0
                    fronts.append([i])
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            front_idx[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            
            return fronts, front_idx

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return [np.inf] * n
            
            # Map front indices to local indices for sorting
            # We need to sort the front by each objective
            dist = [0.0] * n
            
            for m in range(self.n_obj):
                # Sort indices by objective m
                sorted_indices = sorted(front_indices, key=lambda i: F[i][m])
                dist[sorted_indices[0]] = np.inf
                dist[sorted_indices[-1]] = np.inf
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                
                if f_max - f_min == 0:
                    continue
                    
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    dist[idx] += (F[sorted_indices[i+1]][m] - F[sorted_indices[i-1]][m]) / (f_max - f_min)
            
            return dist

        def tournament_select(pop_F, ranks, n_select=2):
            best_idx = 0
            best_rank = ranks[0]
            for _ in range(n_select):
                idx = np.random.randint(len(pop_F))
                if ranks[idx] < best_rank:
                    best_rank = ranks[idx]
                    best_idx = idx
            return best_idx

        def sbx_crossover(x1, x2, lo, hi, eta=20):
            x = x1.copy()
            for k in range(self.dim):
                if np.random.random() < 0.5:
                    if abs(x2[k] - x1[k]) < 1e-14:
                        continue
                    if np.random.random() < 0.5:
                        r = np.random.random()
                        alpha = 1 + 2 * (x1[k] - lo[k]) / (x2[k] - x1[k])
                        beta = 1 + 2 * (hi[k] - x2[k]) / (x2[k] - x1[k])
                        if r <= 1 / (alpha + 1):
                            mul = (r * (alpha + 1)) ** (1 / (eta + 1))
                        else:
                            mul = (1 / (2 - r * (alpha + 1))) ** (1 / (eta + 1))
                        x[k] = 0.5 * ((x1[k] + x2[k]) - mul * (x2[k] - x1[k]))
                    else:
                        r = np.random.random()
                        alpha = 1 + 2 * (x2[k] - lo[k]) / (x2[k] - x1[k])
                        beta = 1 + 2 * (hi[k] - x1[k]) / (x2[k] - x1[k])
                        if r <= 1 / (beta + 1):
                            mul = (r * (beta + 1)) ** (1 / (eta + 1))
                        else:
                            mul = (1 / (2 - r * (beta + 1))) ** (1 / (eta + 1))
                        x[k] = 0.5 * ((x1[k] + x2[k]) + mul * (x2[k] - x1[k]))
            return np.clip(x, lo, hi)

        def polynomial_mutate(x, lo, hi, eta=20):
            for k in range(self.dim):
                if np.random.random() < 0.1: # Mutation rate 0.1
                    y = x[k]
                    if abs(y - lo[k]) < 1e-14 or abs(hi[k] - y) < 1e-14:
                        continue
                    r = np.random.random()
                    if y < (lo[k] + hi[k]) / 2:
                        xy = 1 + (y - lo[k]) / (hi[k] - lo[k])
                        delta = (2 * xy) ** (eta + 1) - 1
                        if r <= 0.5 / (delta + 1):
                            mul = (2 * r) ** (1 / (eta + 1))
                        else:
                            mul = (1 / (2 - 2 * r)) ** (1 / (eta + 1))
                        y = y + (hi[k] - lo[k]) * (mul - 1)
                    else:
                        xy = 1 + (hi[k] - y) / (hi[k] - lo[k])
                        delta = (2 * xy) ** (eta + 1) - 1
                        if r <= 0.5 / (delta + 1):
                            mul = (2 * (1 - r)) ** (1 / (eta + 1))
                        else:
                            mul = (1 / (2 - 2 * (1 - r))) ** (1 / (eta + 1))
                        y = y - (hi[k] - lo[k]) * (mul - 1)
                    x[k] = np.clip(y, lo[k], hi[k])
            return x

        # Main loop
        while used + pop_size <= self.budget:
            # Create offspring
            Q_X = []
            Q_F = []
            for _ in range(pop_size):
                p1_idx = tournament_select(F, *fast_non_dominated_sort(F)[1] if False else (F, [0]*pop_size)) # Placeholder, need ranks
                
            # Let's restructure to calculate ranks once per generation
            fronts, ranks = fast_non_dominated_sort(F)
            
            Q_X = []
            for _ in range(pop_size):
                p1_idx = tournament_select(F, ranks)
                p2_idx = tournament_select(F, ranks)
                
                child = sbx_crossover(X[p1_idx], X[p2_idx], lo, hi)
                child = polynomial_mutate(child, lo, hi)
                Q_X.append(child)
                
            Q_F = np.array([problem(x) for x in Q_X])
            used += pop_size
            
            # Combine
            R_X = np.vstack([X, Q_X])
            R_F = np.vstack([F, Q_F])
            
            # Sort
            fronts, ranks = fast_non_dominated_sort(R_F)
            
            # Select next P
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                else:
                    # Need to select from this front based on crowding
                    # Calculate crowding distance for this front
                    # We need to pass the indices of the front to crowding distance
                    # But crowding distance function above takes front_indices and F
                    # Let's adapt
                    front_F = R_F[front]
                    # We need to map local indices back to global?
                    # My crowding_distance function assumes front_indices are indices into F.
                    # So we pass `front` and `R_F`.
                    cd = crowding_distance(R_F, front)
                    # Sort front by cd descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
        # If we have budget left, we can just stop or add random points.
        # The loop stops when used + pop_size > budget.
        # We might have used < budget.
        # Let's just return X. The evaluator records all problem() calls.
        
        return X
# EVOLVE-BLOCK-END
