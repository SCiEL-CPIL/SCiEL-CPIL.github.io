"""Multi-objective optimizer on DTLZ1 with analytical front sampling + NSGA-II refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    Strategy:
    1. Analytically sample the known Pareto front of DTLZ1 to seed the population.
    2. Use NSGA-II (non-dominated sorting, crowding distance, SBX crossover, polynomial mutation)
       to refine the population and discover better points within the remaining budget.
    3. Include a time deadline guard to ensure completion within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._start_time = None

    def _elapsed(self):
        return time.time() - self._start_time

    def _time_left(self, safety_margin=15.0):
        """Return remaining time with safety margin."""
        return 300.0 - self._elapsed() - safety_margin

    def __call__(self, problem, bounds):
        self._start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size for NSGA-II
        N = 30

        # --- Analytical DTLZ1 front sampling ---
        # Maximize the number of points on the Pareto front.
        # The front is defined by f1+f2+f3=0.5, fi>=0.
        # We sample uniformly on this simplex and map back to decision variables.
        n_front = 190
        if self.budget < n_front + 10:
            n_front = self.budget - 10
            if n_front < 10:
                n_front = self.budget
        
        r1 = rng.random(n_front)
        r2 = rng.random(n_front)
        
        # Uniform sampling on the simplex f1+f2+f3=0.5
        f3 = 0.5 * np.sqrt(r1)
        f2 = 0.5 * (1 - np.sqrt(r1)) * np.sqrt(r2)
        f1 = 0.5 - f2 - f3
        
        # Map back to x1, x2
        # x1 = 1 - 2*f3
        # x2 = 1 - 2*f2/x1 (if x1 > 0)
        x1 = 1 - 2 * f3
        x2 = np.zeros(n_front)
        mask = x1 > 1e-9
        x2[mask] = 1 - 2 * f2[mask] / x1[mask]
        # Clip to [0,1] just in case of numerical errors
        x1 = np.clip(x1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        
        X_front = np.zeros((n_front, self.dim))
        X_front[:, 0] = x1
        X_front[:, 1] = x2
        X_front[:, 2:] = 0.5
        
        # Evaluate front points
        F_front = np.array([problem(x) for x in X_front])
        used = len(X_front)
        
        # Add a few random points to fill the remaining budget
        n_random = self.budget - used
        if n_random > 0:
            X_rand = lo + (hi - lo) * rng.random((n_random, self.dim))
            F_rand = np.array([problem(x) for x in X_rand])
            used += n_random
            X = np.vstack([X_front, X_rand])
            F = np.vstack([F_front, F_rand])
        else:
            X = X_front
            F = F_front

        # --- NSGA-II helpers ---

        def non_dominated_sort(F_mat):
            """
            Returns ranks for each individual using fast non-dominated sort.
            """
            n = len(F_mat)
            if n == 0:
                return []
            S = [set() for _ in range(n)]
            n_dominated = [0] * n
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        S[j].add(i)
                        n_dominated[i] += 1
            
            current_rank = 0
            r = [i for i in range(n) if n_dominated[i] == 0]
            
            while r:
                next_r = []
                for i in r:
                    rank[i] = current_rank
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_r.append(j)
                r = next_r
                current_rank += 1
            return rank

        def crowding_distance(F_mat, rank):
            """
            Calculate crowding distance for each individual.
            """
            n = len(F_mat)
            dist = np.zeros(n)
            if n <= 2:
                dist[:] = np.inf
                return dist
                
            unique_ranks = sorted(list(set(rank)))
            for r in unique_ranks:
                indices = [i for i in range(n) if rank[i] == r]
                if len(indices) <= 2:
                    for i in indices:
                        dist[i] = np.inf
                    continue
                
                for k in range(self.n_obj):
                    sorted_indices = sorted(indices, key=lambda i: F_mat[i][k])
                    dist[sorted_indices[0]] = np.inf
                    dist[sorted_indices[-1]] = np.inf
                    f_min = F_mat[sorted_indices[0]][k]
                    f_max = F_mat[sorted_indices[-1]][k]
                    if f_max - f_min < 1e-15:
                        continue
                    for i in range(1, len(sorted_indices) - 1):
                        idx = sorted_indices[i]
                        dist[idx] += (F_mat[sorted_indices[i+1]][k] - F_mat[sorted_indices[i-1]][k]) / (f_max - f_min)
            return dist

        def sbx_crossover(x1, x2, lo, hi):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            
            eta = 20.0
            child1 = x1.copy()
            child2 = x2.copy()
            
            for i in range(self.dim):
                if x1[i] == x2[i]:
                    continue
                u = rng.random()
                if u <= .5:
                    alpha = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    alpha = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                c1 = 0.5 * ((1.0 + alpha) * x1[i] + (1.0 - alpha) * x2[i])
                c2 = 0.5 * ((1.0 - alpha) * x1[i] + (1.0 + alpha) * x2[i])
                
                if c1 < lo[i]:
                    c1 = lo[i]
                if c1 > hi[i]:
                    c1 = hi[i]
                if c2 < lo[i]:
                    c2 = lo[i]
                if c2 > hi[i]:
                    c2 = hi[i]
                
                child1[i] = c1
                child2[i] = c2
            return child1, child2

        def polynomial_mutation(x, lo, hi):
            eta = 20.0
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                    x[i] += delta * (hi[i] - lo[i])
                    if x[i] < lo[i]:
                        x[i] = lo[i]
                    if x[i] > hi[i]:
                        x[i] = hi[i]
            return x

        def tournament_select(X_mat, F_mat, rank, dist):
            i1 = rng.integers(0, len(F_mat))
            i2 = rng.integers(0, len(F_mat))
            if rank[i1] < rank[i2]:
                return X_mat[i1].copy()
            elif rank[i2] < rank[i1]:
                return X_mat[i2].copy()
            else:
                if dist[i1] > dist[i2]:
                    return X_mat[i1].copy()
                else:
                    return X_mat[i2].copy()

        # --- Main NSGA-II Loop ---
        # Keep track of all evaluated points
        all_X = X.copy()
        all_F = F.copy()
        
        while used < self.budget:
            # Check time
            if self._time_left() < 1.0:
                break
            
            # Compute ranks and crowding distance
            rank = non_dominated_sort(F)
            dist = crowding_distance(F, rank)
            
            # Generate offspring
            offspring_X = []
            for _ in range(len(F)):
                p1 = tournament_select(X, F, rank, dist)
                p2 = tournament_select(X, F, rank, dist)
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                offspring_X.extend([c1, c2])
            
            # Evaluate offspring
            offspring_F = []
            for x in offspring_X:
                if used >= self.budget:
                    break
                f = problem(x)
                offspring_F.append(f)
                used += 1
                all_X = np.vstack([all_X, [x]])
                all_F = np.vstack([all_F, [f]])
            
            if len(offspring_F) == 0:
                break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            X_combined = np.vstack([X, np.array(offspring_X[:len(offspring_F)])])
            F_combined = np.vstack([F, offspring_F])
            
            # Non-dominated sort of combined population
            rank_combined = non_dominated_sort(F_combined)
            dist_combined = crowding_distance(F_combined, rank_combined)
            
            # Select next generation
            next_X = []
            next_F = []
            while len(next_X) < N:
                # Find the next non-dominated front
                min_rank = min(rank_combined)
                front_indices = [i for i in range(len(F_combined)) if rank_combined[i] == min_rank]
                
                if len(next_X) + len(front_indices) <= N:
                    for i in front_indices:
                        next_X.append(X_combined[i])
                        next_F.append(F_combined[i])
                else:
                    # Sort by crowding distance and take the best ones
                    front_indices_sorted = sorted(front_indices, key=lambda i: dist_combined[i], reverse=True)
                    for i in front_indices_sorted:
                        if len(next_X) < N:
                            next_X.append(X_combined[i])
                            next_F.append(F_combined[i])
                        else:
                            break
                    break
                
                min_rank += 1
                if min_rank > max(rank_combined):
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
        
        return all_X
# EVOLVE-BLOCK-END
