import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_vec = hi - lo

        # Population size
        pop_size = 20
        # Initial population
        X = lo + range_vec * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Fast non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]  # S[i] = set of indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of solutions dominating i
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            
            front = set()
            for i in range(n):
                if n_dom[i] == 0:
                    front.add(i)
                    fronts.append(front)
            
            while front:
                next_front = set()
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.add(j)
                front = next_front
                if front:
                    fronts.append(front)
            
            return fronts

        # Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            m = self.n_obj
            n = len(indices)
            cd = {i: 0.0 for i in indices}
            
            for k in range(m):
                sorted_idx = sorted(indices, key=lambda i: F[i][k])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                
                f_min = F[sorted_idx[0]][k]
                f_max = F[sorted_idx[-1]][k]
                f_range = f_max - f_min
                if f_range < 1e-10:
                    continue
                
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F[sorted_idx[i + 1]][k] - F[sorted_idx[i - 1]][k]) / f_range
            
            return cd

        # Binary tournament selection
        def tournament_select(F, fronts, cd):
            i = rng.integers(len(F))
            j = rng.integers(len(F))
            
            # Compare by front
            front_i = next(fi for fi, front in enumerate(fronts) if i in front)
            front_j = next(fi for fi, front in enumerate(fronts) if j in front)
            
            if front_i < front_j:
                return i
            elif front_j < front_i:
                return j
            else:
                # Same front, use crowding distance
                if cd[i] > cd[j]:
                    return i
                else:
                    return j

        # SBX crossover
        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            
            for i in range(self.dim):
                if rng.random() < 0.5:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                else:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1 / (eta + 1))
                        
                        c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        
                        child1[i] = c1
                        child2[i] = c2
            
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        # Polynomial mutation
        def polynomial_mutation(x, eta=20.0):
            x_mut = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    
                    range_i = hi[i] - lo[i]
                    x_mut[i] = x[i] + delta * range_i
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            
            return x_mut

        # Main loop
        while used < self.budget:
            # Compute fronts and crowding distances
            fronts = fast_non_dominated_sort(F)
            cd = {}
            for front in fronts:
                front_list = list(front)
                front_cd = crowding_distance(F, front_list)
                cd.update(front_cd)

            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and used + 2 <= self.budget:
                # Select parents
                p1 = tournament_select(F, fronts, cd)
                p2 = tournament_select(F, fronts, cd)
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1], X[p2])
                
                # Mutation
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                used += 2
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)
            
            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            fronts = fast_non_dominated_sort(combined_F)
            
            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                front_list = list(front)
                if len(new_X) + len(front_list) <= pop_size:
                    for idx in front_list:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    front_cd = crowding_distance(combined_F, front_list)
                    sorted_front = sorted(front_list, key=lambda i: front_cd[i], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        return X
