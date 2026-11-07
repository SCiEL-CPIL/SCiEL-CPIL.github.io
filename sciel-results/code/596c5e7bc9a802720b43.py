"""Multi-objective optimizer on RE34: Hypervolume-indicator based evolutionary strategy (SPEA2-style)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import heapq


class Optimizer:
    """SPEA2-style optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        X = []
        F = []

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            used += 1
            return x, f

        # SPEA2 parameters
        # Archive size K is typically 2*N
        # Let's use N=20 for the mating pool, so Archive K=40.
        N = 20
        K = 40

        rng = np.random.default_rng(42)

        # Initialize population
        for _ in range(N):
            x = rng.uniform(lo, hi)
            x, f = evaluate(x)
            X.append(x)
            F.append(f)

        X = np.array(X)
        F = np.array(F)

        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    if not fronts:
                        fronts.append([])
                    fronts[-1].append(i)

            while fronts[-1]:
                next_front = []
                for i in fronts[-1]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                else:
                    break

            # Remove the last empty front if any
            if fronts and not fronts[-1]:
                fronts.pop()

            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.ones(len(indices)) * np.inf

            dist = np.zeros(len(indices))
            m = len(indices)

            for k in range(F.shape[1]):
                f_k = F[indices, k]
                f_min = np.min(f_k)
                f_max = np.max(f_k)

                if f_max == f_min:
                    continue

                # Sort by objective k
                order = np.argsort(f_k)
                sorted_indices = [indices[order[i]] for i in range(m)]

                # End points get infinite distance
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf

                # Normalize by range
                for i in range(1, m - 1):
                    d = (f_k[order[i + 1]] - f_k[order[i - 1]]) / (f_max - f_min)
                    dist[order[i]] += d

            return dist

        # Main loop
        gen = 0
        while used < budget:
            if time.time() > deadline:
                break

            # 1. Combine population and archive?
            # In SPEA2, we usually maintain an archive of non-dominated solutions.
            # Here, let's just keep the best N solutions from the current generation
            # plus any new non-dominated solutions found, up to size K.
            # But to keep it simple and fast, we'll use a standard evolutionary step:
            # Generate N offspring, combine with parents (2N), select best N.

            # Generate Offspring
            X_off = []
            F_off = []

            for _ in range(N):
                # Tournament selection or random selection?
                # Let's use binary tournament selection based on fitness.
                # Fitness in SPEA2:
                # 1. Non-dominated solutions have lower fitness than dominated.
                # 2. Among non-dominated, lower crowding distance is better? No, higher is better for diversity.
                # Actually, SPEA2 fitness:
                #   R(i) = number of individuals dominated by i
                #   Fitness(i) = R(i) + 1 (if i is non-dominated, R(i) is count of dominated)
                #   Wait, standard SPEA2:
                #   Fitness(i) = R(i) + 1 for non-dominated? No.
                #   Let's stick to simple NSGA-II style selection: Non-dominated sorting + Crowding distance.
                #   This is effectively SPEA2-like if we use a small population and good diversity.

                # Select 2 parents
                # We need a ranking. Let's compute ranks for current X, F.
                fronts = non_dominated_sort(F)
                ranks = np.zeros(len(F), dtype=int)
                for rank, front in enumerate(fronts):
                    for idx in front:
                        ranks[idx] = rank

                # Crowding distances for each front
                cd = np.zeros(len(F))
                for front in fronts:
                    cd[front] = crowding_distance(F, front)

                # Fitness: lower rank is better, higher cd is better.
                # Tournament: pick 2 random, choose better one.
                def better(i, j):
                    if ranks[i] != ranks[j]:
                        return ranks[i] < ranks[j]
                    return cd[i] > cd[j]

                p1, p2 = rng.choice(N, size=2, replace=False)
                if better(p1, p2):
                    p1, p2 = p2, p1

                # SBX Crossover
                x1 = X[p1]
                x2 = X[p2]
                x_child = np.zeros(dim)

                eta = 20.0
                for d in range(dim):
                    if rng.random() < 0.5:
                        if abs(x1[d] - x2[d]) > 1e-14:
                            if rng.random() <= 1.0 / (eta + 1.0):
                                beta = (2.0 * rng.random()) ** (1.0 / (eta + 1.0))
                            else:
                                beta = (1.0 / (2.0 * (1.0 - rng.random()))) ** (1.0 / (eta + 1.0))

                            x_child[d] = 0.5 * ((1.0 + beta) * x1[d] + (1.0 - beta) * x2[d])
                        else:
                            x_child[d] = x1[d]
                    else:
                        if abs(x2[d] - x1[d]) > 1e-14:
                            if rng.random() <= 1.0 / (eta + 1.0):
                                beta = (2.0 * rng.random()) ** (1.0 / (eta + 1.0))
                            else:
                                beta = (1.0 / (2.0 * (1.0 - rng.random()))) ** (1.0 / (eta + 1.0))

                            x_child[d] = 0.5 * ((1.0 + beta) * x2[d] + (1.0 - beta) * x1[d])
                        else:
                            x_child[d] = x2[d]

                # Mutation
                # Polynomial mutation
                eta_m = 20.0
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        delta = 1.0 - (x_child[d] - lo[d]) / (hi[d] - lo[d] + 1e-14)
                        delta_q = 2.0 * delta - 1.0
                        if rng.random() <= delta_q / (2.0 * (1.0 - delta)):
                            u = rng.random()
                            delta = (1.0 + (1.0 - delta) * u) ** (1.0 / (eta_m + 1.0))
                            x_child[d] = x_child[d] + (hi[d] - x_child[d]) * (delta - 1.0)
                        else:
                            u = rng.random()
                            delta = (1.0 + delta * u) ** (1.0 / (eta_m + 1.0))
                            x_child[d] = x_child[d] - (x_child[d] - lo[d]) * (1.0 - delta)

                x_child = np.clip(x_child, lo, hi)
                x_child, f_child = evaluate(x_child)
                X_off.append(x_child)
                F_off.append(f_child)

                if used >= budget:
                    break

            if used >= budget:
                break

            X_off = np.array(X_off)
            F_off = np.array(F_off)

            # Combine
            X_R = np.vstack([X, X_off])
            F_R = np.vstack([F, F_off])

            # Selection: Keep best N
            fronts_R = non_dominated_sort(F_R)

            X_next = []
            F_next = []
            selected_indices = []

            for front in fronts_R:
                if len(selected_indices) + len(front) <= N:
                    selected_indices.extend(front)
                else:
                    remaining = N - len(selected_indices)
                    # Crowding distance within this front
                    cd_front = crowding_distance(F_R, front)
                    # Sort by cd descending
                    order = np.argsort(-cd_front)
                    for i in order[:remaining]:
                        selected_indices.append(front[i])
                    break

            X = X_R[selected_indices]
            F = F_R[selected_indices]

            gen += 1

        return X
# EVOLVE-BLOCK-END
