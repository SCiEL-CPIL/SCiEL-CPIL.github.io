import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        pop_size = 20

        # Latin Hypercube-style initialization for better coverage
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(pop_size)
            X[perm, d] = np.linspace(0, 1, pop_size) + np.random.uniform(0, 1/pop_size, pop_size)
            X[:, d] = np.clip(X[:, d], 0, 1)
        X = lo + X * range_
        F = np.array([problem(x) for x in X])
        evals = pop_size

        # Helper: Non-dominated sort
        def fast_non_dominated_sort(F_arr):
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated = [[] for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[j].append(i)
                        domination_count[i] += 1

            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F_arr, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            distances = np.zeros(n)
            for m in range(self.n_obj):
                f_vals = F_arr[front_indices, m]
                min_val = f_vals.min()
                max_val = f_vals.max()
                if max_val - min_val < 1e-15:
                    continue
                sorted_idx = np.argsort(f_vals)
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    distances[sorted_idx[k]] += (f_vals[sorted_idx[k + 1]] - f_vals[sorted_idx[k - 1]]) / (max_val - min_val)
            return distances

        # Helper: Tournament selection
        def tournament_select(F_arr, fronts, crowding_dists):
            rank = np.zeros(len(F_arr), dtype=int)
            cd = np.zeros(len(F_arr))
            for r, front in enumerate(fronts):
                cd_front = crowding_dists[r]
                for i, idx in enumerate(front):
                    rank[idx] = r
                    cd[idx] = cd_front[i]

            idx1 = np.random.randint(len(F_arr))
            idx2 = np.random.randint(len(F_arr))

            if rank[idx1] < rank[idx2]:
                return idx1
            elif rank[idx2] < rank[idx1]:
                return idx2
            else:
                return idx1 if cd[idx1] > cd[idx2] else idx2

        # SBX crossover
        def sbx_crossover(x1, x2, lo, hi, eta_c=20.0):
            child = np.zeros_like(x1)
            for i in range(len(x1)):
                if x1[i] != x2[i]:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta_c + 1))
                        c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    else:
                        beta = (2 * (1 - u)) ** (1 / (eta_c + 1))
                        c1 = 0.5 * ((1 + beta) * x2[i] + (1 - beta) * x1[i])
                        c2 = 0.5 * ((1 - beta) * x2[i] + (1 + beta) * x1[i])
                    child[i] = c1
                else:
                    child[i] = x1[i]
            return np.clip(child, lo, hi)

        # Polynomial mutation
        def poly_mutate(x, lo, hi, eta_m=20.0, prob=0.3):
            child = x.copy()
            for i in range(len(x)):
                if np.random.random() < prob:
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    child[i] = x[i] + delta * (hi[i] - lo[i])
            return np.clip(child, lo, hi)

        # Main Loop
        while evals < self.budget:
            if time.time() > deadline:
                break

            # Calculate fronts and crowding for current population
            fronts = fast_non_dominated_sort(F)
            crowding_dists = []
            for front in fronts:
                crowding_dists.append(crowding_distance(F, front))

            # Select two parents
            parent1_idx = tournament_select(F, fronts, crowding_dists)
            parent2_idx = tournament_select(F, fronts, crowding_dists)
            if parent1_idx == parent2_idx:
                parent2_idx = (parent2_idx + 1) % len(F)

            parent1 = X[parent1_idx]
            parent2 = X[parent2_idx]

            # Crossover and mutation
            child = sbx_crossover(parent1, parent2, lo, hi, eta_c=15.0)
            child = poly_mutate(child, lo, hi, eta_m=15.0, prob=0.2)

            fc = problem(child)
            evals += 1

            # Combine population and child
            X_new = np.vstack([X, child.reshape(1, -1)])
            F_new = np.vstack([F, fc.reshape(1, -1)])

            # Select next population
            fronts_new = fast_non_dominated_sort(F_new)
            crowding_dists_new = []
            for front in fronts_new:
                crowding_dists_new.append(crowding_distance(F_new, front))

            # Fill next population
            X_next = []
            F_next = []
            for i, front in enumerate(fronts_new):
                if len(X_next) + len(front) <= pop_size:
                    for idx in front:
                        X_next.append(X_new[idx])
                        F_next.append(F_new[idx])
                else:
                    remaining = pop_size - len(X_next)
                    cd_front = crowding_dists_new[i]
                    sorted_indices = np.argsort(-cd_front)
                    for k in range(remaining):
                        idx = front[sorted_indices[k]]
                        X_next.append(X_new[idx])
                        F_next.append(F_new[idx])
                    break

            X = np.array(X_next)
            F = np.array(F_next)

            # Local refinement: if we have time, do a local search on the best point
            if evals < self.budget and time.time() < deadline - 10.0:
                # Find the point with best minimum objective (proxy for Pareto optimality)
                min_obj = np.min(F, axis=1)
                best_idx = np.argmin(min_obj)
                best_x = X[best_idx].copy()
                best_f = F[best_idx].copy()
                
                # Simple local search with decreasing step size
                step_sizes = [0.1, 0.05, 0.02, 0.01]
                for ss in step_sizes:
                    if evals >= self.budget or time.time() > deadline:
                        break
                    improved = False
                    for _ in range(3):  # Try a few random perturbations
                        if evals >= self.budget or time.time() > deadline:
                            break
                        perturbation = np.random.randn(self.dim) * ss * range_
                        candidate = np.clip(best_x + perturbation, lo, hi)
                        cand_f = problem(candidate)
                        evals += 1
                        # Accept if it improves the minimum objective or is non-dominated
                        if np.min(cand_f) < np.min(best_f) - 1e-10:
                            best_x = candidate
                            best_f = cand_f
                            improved = True
                        elif np.all(cand_f <= best_f) and np.any(cand_f < best_f):
                            best_x = candidate
                            best_f = cand_f
                            improved = True
                    if not improved:
                        break

        return X
