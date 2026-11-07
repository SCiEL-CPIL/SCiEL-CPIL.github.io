# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II for multi-objective optimization (DTLZ4)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = 60
        if self.budget < pop_size * 2:
            pop_size = max(10, self.budget // 3)

        # Initialize population
        X = lo + span * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II parameters
        crossover_rate = 0.9
        mutation_rate = 1.0 / dim
        sigma = 0.5

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # Domination count
            n_dom = np.zeros(n, dtype=int)  # Number of solutions dominated by i
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            current_front = [i for i in range(n) if n_dom[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            return fronts

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = sorted(front_indices, key=lambda i: F[i, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F[sorted_idx[-1], m] - F[sorted_idx[0], m]
                if f_range > 1e-10:
                    for k in range(1, n - 1):
                        cd[sorted_idx[k]] += (F[sorted_idx[k + 1], m] - F[sorted_idx[k - 1], m]) / f_range
            return cd

        def tournament_select(F, fronts, k=2):
            """Tournament selection based on non-dominated rank and crowding distance."""
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            
            # Calculate crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                cd[front] = crowding_distance(F, front)
            
            # Tournament
            candidates = np.random.choice(len(F), size=k, replace=False)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best] or (rank[c] == rank[best] and cd[c] > cd[best]):
                    best = c
            return best

        def sbx_crossover(x1, x2, eta=30.0):
            """Simulated Binary Crossover."""
            if np.random.random() > crossover_rate:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            for k in range(dim):
                if x1[k] == x2[k]:
                    continue
                if np.random.random() < 0.5:
                    if x1[k] < x2[k]:
                        x_lo, x_hi = x1[k], x2[k]
                    else:
                        x_lo, x_hi = x2[k], x1[k]
                    
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    
                    c1[k] = 0.5 * ((1 + beta) * x_lo + (1 - beta) * x_hi)
                    c2[k] = 0.5 * ((1 - beta) * x_lo + (1 + beta) * x_hi)
            return c1, c2

        def polynomial_mutation(x, eta=20.0):
            """Polynomial mutation."""
            x = x.copy()
            for k in range(dim):
                if np.random.random() < mutation_rate:
                    x_lo = lo[k]
                    x_hi = hi[k]
                    if x_hi - x_lo < 1e-10:
                        continue
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x[k] = x[k] + delta * (x_hi - x_lo)
            return np.clip(x, lo, hi)

        while used < self.budget:
            # Create offspring population
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and used < self.budget:
                # Select parents via tournament
                p1 = tournament_select(F, fast_non_dominated_sort(F))
                p2 = tournament_select(F, fast_non_dominated_sort(F))
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1], X[p2])
                
                # Mutation
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used >= self.budget:
                    break
                
                f2 = problem(c2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)
            
            if len(offspring_X) == 0:
                break
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(combined_F)
            
            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
