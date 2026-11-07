"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with adaptive mutation and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: use more of the budget for better exploration
        pop_size = min(100, max(20, self.budget // 2))
        
        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            fronts = []
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            nd = np.zeros(n, dtype=int)  # nd[i] = number of points dominating i
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        nd[i] += 1
            
            # First front
            front = [i for i in range(n) if nd[i] == 0]
            ranks[front] = 0
            fronts.append(front)
            
            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                            ranks[j] = k + 1
                k += 1
                front = next_front
                if front:
                    fronts.append(front)
            
            return ranks, fronts

        # Crowding distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            m = self.n_obj
            for k in range(m):
                obj_vals = F[indices, k]
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                if obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]] > 1e-10:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (obj_vals[sorted_idx[i+1]] - obj_vals[sorted_idx[i-1]]) / (obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]])
            return cd

        # Selection: binary tournament
        def tournament_select(pop, F, ranks, cd):
            i = rng.integers(len(pop))
            j = rng.integers(len(pop))
            if ranks[i] < ranks[j]:
                return i
            elif ranks[j] < ranks[i]:
                return j
            else:
                return i if cd[i] > cd[j] else j

        # Generate offspring
        def generate_offspring(X, F, ranks, cd):
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                # Select parents
                p1 = tournament_select(X, F, ranks, cd)
                p2 = tournament_select(X, F, ranks, cd)
                
                # Simulated binary crossover (SBX)
                if rng.random() < 0.9:
                    child1 = np.empty(self.dim)
                    child2 = np.empty(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            x1, x2 = X[p1, d], X[p2, d]
                            if abs(x1 - x2) > 1e-10:
                                beta1 = 1 + 2 * min((x1 - lo[d]) / (abs(x1 - x2) + 1e-10), 1)
                                beta2 = 1 + 2 * ((hi[d] - x2) / (abs(x1 - x2) + 1e-10))
                                eta1 = 1 / (beta1 + 1) if rng.random() <= 1 / (beta1 + 1) else (2 - beta1) ** (-1 / (beta1 + 1))
                                eta2 = 1 / (beta2 + 1) if rng.random() <= 1 / (beta2 + 1) else (2 - beta2) ** (-1 / (beta2 + 1))
                                child1[d] = 0.5 * ((1 + eta1) * x1 + (1 - eta1) * x2)
                                child2[d] = 0.5 * ((1 + eta2) * x2 + (1 - eta2) * x1)
                            else:
                                child1[d] = x1
                                child2[d] = x2
                        else:
                            child1[d] = X[p1, d]
                            child2[d] = X[p2, d]
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                else:
                    offspring_X.append(X[p1].copy())
                    offspring_X.append(X[p2].copy())
            
            # Mutate
            mutation_rate = 1.0 / self.dim
            for i in range(len(offspring_X)):
                for d in range(self.dim):
                    if rng.random() < mutation_rate:
                        # Polynomial mutation
                        x = offspring_X[i][d]
                        delta1 = (x - lo[d]) / (hi[d] - lo[d] + 1e-10)
                        delta2 = (hi[d] - x) / (hi[d] - lo[d] + 1e-10)
                        beta = 1 + 2 * min(delta1, delta2)
                        alpha = 2 + (beta + 1) * 20
                        u = rng.random()
                        if u < 0.5:
                            xy = (1 + (x - lo[d]) / (hi[d] - lo[d] + 1e-10)) * (2 * u) ** (1 / (alpha + 1)) - 1
                        else:
                            xy = 1 - (1 + (hi[d] - x) / (hi[d] - lo[d] + 1e-10)) * (2 * (1 - u)) ** (1 / (alpha + 1))
                        offspring_X[i][d] = x + (hi[d] - lo[d]) * xy
                        offspring_X[i][d] = np.clip(offspring_X[i][d], lo[d], hi[d])
            
            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array([problem(x) for x in offspring_X])
            return offspring_X, offspring_F

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Evaluate offspring
            offspring_X, offspring_F = generate_offspring(X, F, *non_dominated_sort(F))
            used += pop_size
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort combined
            ranks, fronts = non_dominated_sort(combined_F)
            
            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need crowding distance to fill remaining spots
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X[:pop_size])
            F = np.array(new_F[:pop_size])

        return X
# EVOLVE-BLOCK-END
