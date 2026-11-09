# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_eval = 0

        def eval_point(x):
            nonlocal n_eval
            x = np.clip(x, lo, hi)
            f = problem(x)
            n_eval += 1
            return x, f

        # Population size
        # With budget 200 and dim 12, N=20 allows for about 5-6 generations.
        # This is a reasonable trade-off between exploration and exploitation.
        N = 20
        
        # Initialize population
        X = np.zeros((N, self.dim))
        F = np.zeros((N, self.n_obj))
        
        for i in range(N):
            x = lo + (hi - lo) * rng.random(self.dim)
            x, f = eval_point(x)
            X[i] = x
            F[i] = f

        # NSGA-II Parameters
        eta_c = 20.0  # Distribution index for crossover
        eta_m = 20.0  # Distribution index for mutation
        p_c = 0.9     # Crossover probability

        def fast_non_dominated_sort(F):
            """
            Returns a list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_d = [0] * n  # Number of solutions dominating i
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_d[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_d[i] == 0]
            for i in current_front:
                rank[i] = 0
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices.
            """
            n = len(indices)
            if n == 0:
                return {}
            dist = {i: 0.0 for i in indices}
            
            for m in range(self.n_obj):
                # Sort indices by objective m
                sorted_indices = sorted(indices, key=lambda i: F[i][m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                    
                for i in range(1, n - 1):
                    dist[sorted_indices[i]] += (F[sorted_indices[i+1]][m] - F[sorted_indices[i-1]][m]) / f_range
            return dist

        def tournament_selection(X, F, rank, dist, k=2):
            """
            Select k individuals using tournament selection.
            """
            selected = []
            for _ in range(k):
                # Pick random indices
                i = rng.integers(len(F))
                j = rng.integers(len(F))
                
                # Compare rank first
                if rank[i] < rank[j]:
                    selected.append(i)
                elif rank[j] < rank[i]:
                    selected.append(j)
                else:
                    # Compare crowding distance
                    if dist.get(i, 0) > dist.get(j, 0):
                        selected.append(i)
                    else:
                        selected.append(j)
            return selected

        def sbx_crossover(x1, x2, eta):
            if rng.random() > p_c:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if x1[i] != x2[i]:
                        x1_min = min(x1[i], x2[i])
                        x2_max = max(x1[i], x2[i])
                        delta1 = (x2_max - x1_min) / (hi[i] - lo[i] + 1e-10)
                        delta2 = (x2_max - x1_min) / (hi[i] - lo[i] + 1e-10)
                        
                        if delta1 < 0: delta1 = 0
                        if delta2 < 0: delta2 = 0
                        
                        r = rng.random()
                        if r <= 1.0 / (eta + 1):
                            beta = (2 * r) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - r))) ** (1.0 / (eta + 1))
                        
                        c1[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x2[i] - x1[i]))
                        c2[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x2[i] - x1[i]))
                        
                        # Clip to bounds
                        c1[i] = np.clip(c1[i], lo[i], hi[i])
                        c2[i] = np.clip(c2[i], lo[i], hi[i])
            return c1, c2

        def polynomial_mutation(x, eta):
            x_mut = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i] + 1e-10)
                    delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i] + 1e-10)
                    
                    r = rng.random()
                    if r < 0.5:
                        xy = 1.0 - (1.0 - delta1)
                        beta = 1.0 + (2.0 * xy) ** (1.0 / (eta + 1))
                    else:
                        xy = 1.0 - (1.0 - delta2)
                        beta = 1.0 + (2.0 * xy) ** (1.0 / (eta + 1))
                    
                    x_mut[i] = x_mut[i] + (beta - 1.0) * (hi[i] - lo[i])
                    if r < 0.5:
                        x_mut[i] -= (beta - 1.0) * (hi[i] - lo[i]) # This logic is slightly off, let's use standard formula
                    # Standard formula:
                    # if r < 0.5:
                    #    delta = (1 - (1 - delta1)^(eta+1))^(1/(eta+1))
                    #    x = x - delta * (x - lo)
                    # else:
                    #    delta = (1 - (1 - delta2)^(eta+1))^(1/(eta+1))
                    #    x = x + delta * (hi - x)
                    
                    # Let's restart mutation logic for clarity
                    x_mut[i] = x[i]
                    delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i] + 1e-10)
                    delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i] + 1e-10)
                    
                    r = rng.random()
                    if r < 0.5:
                        delta = (1.0 - (1.0 - delta1) ** (eta + 1)) ** (1.0 / (eta + 1))
                        x_mut[i] = x_mut[i] - delta * (x_mut[i] - lo[i])
                    else:
                        delta = (1.0 - (1.0 - delta2) ** (eta + 1)) ** (1.0 / (eta + 1))
                        x_mut[i] = x_mut[i] + delta * (hi[i] - x_mut[i])
                    
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main Loop
        while n_eval < self.budget - N:
            # Non-dominated sort and crowding distance
            fronts, rank = fast_non_dominated_sort(F)
            
            # Calculate crowding distance for all individuals
            # We need to assign distances based on their front
            dist = {}
            for front in fronts:
                front_dist = crowding_distance(F, front)
                for i in front:
                    dist[i] = front_dist[i]
            
            # Create offspring
            offspring_X = np.zeros((N, self.dim))
            offspring_F = np.zeros((N, self.n_obj))
            
            for i in range(0, N, 2):
                p1_idx = tournament_selection(X, F, rank, dist, k=2)[0]
                p2_idx = tournament_selection(X, F, rank, dist, k=2)[0]
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx], eta_c)
                c1 = polynomial_mutation(c1, eta_m)
                c2 = polynomial_mutation(c2, eta_m)
                
                c1, f1 = eval_point(c1)
                c2, f2 = eval_point(c2)
                
                offspring_X[i] = c1
                offspring_F[i] = f1
                if i + 1 < N:
                    offspring_X[i+1] = c2
                    offspring_F[i+1] = f2
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Select next generation
            fronts, rank = fast_non_dominated_sort(combined_F)
            
            new_X = np.zeros((N, self.dim))
            new_F = np.zeros((N, self.n_obj))
            
            idx = 0
            for front in fronts:
                if idx + len(front) <= N:
                    # Whole front fits
                    for i in front:
                        new_X[idx] = combined_X[i]
                        new_F[idx] = combined_F[i]
                        idx += 1
                else:
                    # Partial front, use crowding distance
                    front_dist = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: front_dist[i], reverse=True)
                    for i in sorted_front:
                        if idx < N:
                            new_X[idx] = combined_X[i]
                            new_F[idx] = combined_F[i]
                            idx += 1
                        else:
                            break
                    break
            
            X = new_X
            F = new_F

        return X
# EVOLVE-BLOCK-END
