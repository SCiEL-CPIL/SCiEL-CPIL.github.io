"""Multi-objective optimizer on WFG9 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective evolutionary optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = min(100, max(20, budget // 2))

        # Initialize population
        rng = np.random.RandomState(42)
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominates = [0] * n
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[i] += 1

            current_front = [i for i in range(n) if n_dominates[i] == 0]
            fronts.append(current_front)

            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            m = len(indices)
            cd = np.zeros(m)
            n_obj = len(F[indices[0]])
            
            for k in range(n_obj):
                obj_vals = F[indices, k]
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                
                min_val = obj_vals[sorted_idx[0]]
                max_val = obj_vals[sorted_idx[-1]]
                range_val = max_val - min_val
                
                if range_val > 1e-15:
                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (obj_vals[sorted_idx[i+1]] - obj_vals[sorted_idx[i-1]]) / range_val
            
            # Map back to original indices
            cd_map = {}
            for i, idx in enumerate(indices):
                cd_map[idx] = cd[i]
            return cd_map

        def tournament_select(fronts, cd_maps, n_select):
            """Tournament selection based on rank and crowding distance."""
            # Assign ranks
            rank = np.zeros(len(fronts[0]) + len(fronts[1]) + len(fronts[2]) if len(fronts) >= 3 else len(fronts[0]) + len(fronts[1]), dtype=int)
            # Actually, let's build a proper rank array for all N individuals
            rank = np.zeros(N, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            
            # Build cd for all individuals (use 0 for those not in first front, but we need cd per front)
            # For simplicity, compute cd for each front separately and store
            all_cd = np.zeros(N)
            for front in fronts:
                if len(front) > 0:
                    cd_map = crowding_distance(F, front)
                    for idx in front:
                        all_cd[idx] = cd_map[idx]
            
            selected = []
            for _ in range(n_select):
                i = rng.randint(N)
                j = rng.randint(N)
                if rank[i] < rank[j]:
                    selected.append(i)
                elif rank[i] > rank[j]:
                    selected.append(j)
                else:
                    if all_cd[i] > all_cd[j]:
                        selected.append(i)
                    else:
                        selected.append(j)
            return selected

        def crossover(x1, x2):
            """Simulated binary crossover."""
            if rng.random() < 0.9:
                eta = 20
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for d in range(dim):
                    if rng.random() < 0.5:
                        if abs(x1[d] - x2[d]) < 1e-14:
                            child1[d] = x1[d]
                            child2[d] = x2[d]
                        else:
                            if rng.random() <= 0.5:
                                beta = (2 * rng.random()) ** (1.0 / (eta + 1))
                                child1[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                                child2[d] = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
                            else:
                                beta = (2 * (1 - rng.random())) ** (1.0 / (eta + 1))
                                child1[d] = 0.5 * ((1 + beta) * x2[d] + (1 - beta) * x1[d])
                                child2[d] = 0.5 * ((1 - beta) * x2[d] + (1 + beta) * x1[d])
                    else:
                        if abs(x1[d] - x2[d]) < 1e-14:
                            child1[d] = x1[d]
                            child2[d] = x2[d]
                        else:
                            if rng.random() <= 0.5:
                                beta = (2 * (1 - rng.random())) ** (1.0 / (eta + 1))
                                child1[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                                child2[d] = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
                            else:
                                beta = (2 * rng.random()) ** (1.0 / (eta + 1))
                                child1[d] = 0.5 * ((1 + beta) * x2[d] + (1 - beta) * x1[d])
                                child2[d] = 0.5 * ((1 - beta) * x2[d] + (1 + beta) * x1[d])
                return np.clip(child1, lo, hi), np.clip(child2, lo, hi)
            else:
                return x1.copy(), x2.copy()

        def mutate(x):
            """Polynomial mutation."""
            eta = 20
            child = x.copy()
            for d in range(dim):
                if rng.random() < 0.9:
                    y = x[d]
                    if y <= lo[d] or y >= hi[d]:
                        child[d] = lo[d] + (hi[d] - lo[d]) * rng.random()
                    else:
                        delta1 = (y - lo[d]) / (hi[d] - lo[d])
                        delta2 = (hi[d] - y) / (hi[d] - lo[d])
                        rand = rng.random()
                        if rand < 0.5:
                            xy = (2 * rand + (1 - 2 * rand) * (delta1 ** (eta + 1))) ** (1.0 / (eta + 1))
                            deltaq = (xy - 1) * (hi[d] - lo[d])
                        else:
                            xy = (2 * (1 - rand) + (2 * rand - 1) * (delta2 ** (eta + 1))) ** (1.0 / (eta + 1))
                            deltaq = (xy - 1) * (hi[d] - lo[d])
                        child[d] = y + deltaq
            return np.clip(child, lo, hi)

        # Main NSGA-II loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < N:
                if time.time() > deadline:
                    break
                
                # Select parents
                parents = tournament_select(fronts, None, 2)
                p1, p2 = parents
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutate(c1)
                c2 = mutate(c2)
                
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used >= budget:
                    break
                
                f2 = problem(c2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)
            
            if len(offspring_X) == 0:
                break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = fast_non_dominated_sort(combined_F)
            
            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= N:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate this front based on crowding distance
                    remaining = N - len(new_X)
                    cd_map = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd_map[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            if len(new_X) == 0:
                break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            # Recompute fronts for next iteration
            fronts = fast_non_dominated_sort(F)

        return X
# EVOLVE-BLOCK-END
