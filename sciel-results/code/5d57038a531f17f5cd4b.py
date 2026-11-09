# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using a hybrid approach:
    1. Latin hypercube sampling for initial spread
    2. Evolutionary search with non-dominated sorting and crowding distance
    3. Local refinement via coordinate descent on promising solutions
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        def time_exceeded():
            return time.time() > deadline

        # Latin Hypercube Sampling for initial population
        pop_size = min(40, self.budget // 5)
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            perm = np.arange(pop_size)
            rng.shuffle(perm)
            X[perm, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(pop_size) + rng.random()) / pop_size

        F = np.array([problem(x) for x in X])
        used = pop_size

        if time_exceeded():
            return X

        # Helper: non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            front = [np.where(np.all(np.sum(F <= F[i], axis=1) == 0, axis=1))[0].tolist()]
            if not front[0]:
                front[0] = list(range(n))
            # Proper non-dominated sort
            fronts = []
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
            current_front = [i for i in range(n) if n_dom[i] == 0]
            fronts.append(current_front)
            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
                k += 1
            return fronts

        # Helper: crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            m = len(indices)
            cd = np.zeros(m)
            for k in range(self.n_obj):
                sorted_idx = np.argsort(F[indices, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                fmin = F[indices, k].min()
                fmax = F[indices, k].max()
                if fmax > fmin:
                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (F[indices[sorted_idx[i + 1]], k] - F[indices[sorted_idx[i - 1]], k]) / (fmax - fmin)
            return cd

        # Helper: binary tournament selection
        def tournament_select(pop_X, pop_F, fronts, cd_map):
            i, j = rng.integers(len(pop_X), size=2)
            fi = fronts.index([x for x in fronts if i in x][0]) if i in [x for front in fronts for x in front] else 0
            fj = fronts.index([x for x in fronts if j in x][0]) if j in [x for x in fronts for x in front] else 0
            # Find front number properly
            front_num = {idx: fnum for fnum, front in enumerate(fronts) for idx in front}
            fi = front_num.get(i, 0)
            fj = front_num.get(j, 0)
            if fi < fj:
                return i
            elif fj < fi:
                return j
            else:
                return i if cd_map.get(i, 0) > cd_map.get(j, 0) else j

        # Helper: crossover
        def crossover(x1, x2):
            child = np.where(rng.random(self.dim) < 0.5, x1, x2)
            return child

        # Helper: mutation
        def mutate(x, sigma):
            x_new = x + sigma * (hi - lo) * rng.standard_normal(self.dim)
            return np.clip(x_new, lo, hi)

        # Main evolutionary loop
        while used < self.budget and not time_exceeded():
            # Non-dominated sort
            fronts = non_dominated_sort(F)
            if not fronts:
                break

            # Calculate crowding distances for first front
            front0 = fronts[0]
            cd_front0 = crowding_distance(F, front0)
            cd_map = {front0[i]: cd_front0[i] for i in range(len(front0))}

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, self.budget - used)

            for _ in range(n_offspring):
                if time_exceeded():
                    break
                p1 = tournament_select(X, F, fronts, cd_map)
                p2 = tournament_select(X, F, fronts, cd_map)
                child = crossover(X[p1], X[p2])
                sigma = 0.1 * (1 - used / self.budget) + 0.01
                child = mutate(child, sigma)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            all_X = np.vstack([X, offspring_X])
            all_F = np.vstack([F, offspring_F])

            # Select next population
            all_fronts = non_dominated_sort(all_F)
            next_X = []
            next_F = []
            for front in all_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(all_X[idx])
                        next_F.append(all_F[idx])
                else:
                    remaining = pop_size - len(next_X)
                    cd_front = crowding_distance(all_F, front)
                    sorted_by_cd = np.argsort(-cd_front)
                    for i in range(remaining):
                        idx = front[sorted_by_cd[i]]
                        next_X.append(all_X[idx])
                        next_F.append(all_F[idx])
                    break

            if next_X:
                X = np.array(next_X)
                F = np.array(next_F)

        # Final local refinement on best points
        if not time_exceeded() and used < self.budget:
            # Take top 5 non-dominated points and refine them
            fronts = non_dominated_sort(F)
            if fronts:
                top_points = fronts[0][:5]
                for idx in top_points:
                    if time_exceeded() or used >= self.budget:
                        break
                    # Coordinate descent
                    x_best = X[idx].copy()
                    f_best = F[idx].copy()
                    for coord in range(self.dim):
                        if time_exceeded() or used >= self.budget:
                            break
                        best_val = f_best.copy()
                        best_x = x_best.copy()
                        for step in np.linspace(-0.1, 0.1, 5):
                            if time_exceeded() or used >= self.budget:
                                break
                            x_try = x_best.copy()
                            x_try[coord] += step * (hi[coord] - lo[coord])
                            x_try = np.clip(x_try, lo, hi)
                            f_try = problem(x_try)
                            used += 1
                            # Check if better in at least one objective without worsening others
                            if np.any(f_try < best_val) and not np.any(f_try > best_val):
                                best_val = f_try
                                best_x = x_try
                        x_best = best_x
                        f_best = best_val
                    # Update if improved
                    if np.any(f_best < F[idx]):
                        X[idx] = x_best
                        F[idx] = f_best

        return X
# EVOLVE-BLOCK-END
