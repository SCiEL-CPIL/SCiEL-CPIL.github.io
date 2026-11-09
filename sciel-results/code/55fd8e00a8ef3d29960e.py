# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Fast and robust multi-objective optimizer for WFG7 (3 objectives, 12 dims)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 290.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # State
        X = []
        F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            fx = problem(x)
            eval_count += 1
            X.append(x)
            F.append(fx)
            return fx

        # Initial random sampling
        n_init = min(30, budget)
        for _ in range(n_init):
            x = lo + span * np.random.rand(dim)
            evaluate(x)

        F_arr = np.array(F)
        X_arr = np.array(X)

        def fast_nondominated_sort(F_arr):
            n = len(F_arr)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        nd[i] += 1
                if nd[i] == 0:
                    rank[i] = 0
                    fronts.append([i])
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return rank, fronts

        def crowding_distance(front, F_arr):
            n = len(front)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                idx = np.argsort(F_arr[front, m])
                dist[front[idx[0]]] = np.inf
                dist[front[idx[-1]]] = np.inf
                fmin = F_arr[front[idx[0]], m]
                fmax = F_arr[front[idx[-1]], m]
                if fmax - fmin > 1e-15:
                    for i in range(1, n - 1):
                        dist[front[idx[i]]] += (F_arr[front[idx[i + 1]], m] - F_arr[front[idx[i - 1]], m]) / (fmax - fmin)
            return dist

        # Main loop
        pop_size = min(40, budget // 2)
        while eval_count < budget and time.time() < deadline:
            if len(F) < pop_size:
                # Keep expanding with random points until we have a decent pop
                x = lo + span * np.random.rand(dim)
                evaluate(x)
                F_arr = np.array(F)
                continue

            # Take a subset for selection (use all if smaller)
            n_sel = min(len(F), pop_size)
            # Use the most recent n_sel points as the working population
            start_idx = len(F) - n_sel
            X_pop = X[start_idx:]
            F_pop = F[start_idx:]
            F_pop_arr = np.array(F_pop)

            # Non-dominated sort
            rank, fronts = fast_nondominated_sort(F_pop_arr)

            # Selection: pick 2 parents from front 0 (or best available)
            if fronts and len(fronts[0]) > 0:
                front0 = fronts[0]
            else:
                front0 = list(range(n_sel))

            # Crowding distance for front 0
            cd = crowding_distance(front0, F_pop_arr)
            # Weighted random selection from front 0 based on crowding
            cd_norm = cd / (cd.sum() + 1e-15)
            parent1_idx = np.random.choice(front0, p=cd_norm)
            parent2_idx = np.random.choice(front0, p=cd_norm)

            p1 = np.array(X_pop[parent1_idx])
            p2 = np.array(X_pop[parent2_idx])

            # SBX crossover
            if np.random.rand() < 0.9:
                eta_c = 15.0
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for i in range(dim):
                    if abs(p1[i] - p2[i]) > 1e-15:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta_c + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                        child1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                        child2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    else:
                        child1[i] = p1[i]
                        child2[i] = p2[i]
            else:
                child1 = p1.copy()
                child2 = p2.copy()

            # Polynomial mutation
            eta_m = 20.0
            for child in [child1, child2]:
                for i in range(dim):
                    if np.random.rand() < 1.0 / dim:
                        u = np.random.rand()
                        if u < 0.5:
                            delta_q = (2 * u) ** (1 / (eta_m + 1)) - 1
                        else:
                            delta_q = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                        child[i] += delta_q * (hi[i] - lo[i])
                child[:] = np.clip(child, lo, hi)

            # Evaluate children
            for child in [child1, child2]:
                if eval_count < budget and time.time() < deadline:
                    evaluate(child)

            # Environmental selection: combine population + offspring, select best pop_size
            # But we keep it simple: just let the population grow and periodically prune
            if len(F) > pop_size * 2:
                # Prune: keep non-dominated + diverse
                F_all_arr = np.array(F)
                rank_all, fronts_all = fast_nondominated_sort(F_all_arr)
                # Keep all of front 0, then fill with others by crowding
                keep = []
                if fronts_all:
                    keep.extend(fronts_all[0])
                    for front in fronts_all[1:]:
                        if len(keep) >= pop_size:
                            break
                        cd_front = crowding_distance(front, F_all_arr)
                        # Keep most crowded
                        order = np.argsort(-cd_front)
                        for idx in order:
                            if len(keep) >= pop_size:
                                break
                            keep.append(front[idx])
                if len(keep) < pop_size:
                    # Fill with random
                    remaining = [i for i in range(len(F)) if i not in keep]
                    np.random.shuffle(remaining)
                    keep.extend(remaining[:pop_size - len(keep)])

                # Rebuild
                keep = list(set(keep))[:pop_size]
                X = [X[i] for i in keep]
                F = [F[i] for i in keep]

        return X
# EVOLVE-BLOCK-END
