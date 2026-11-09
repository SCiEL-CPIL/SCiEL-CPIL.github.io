"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: balance exploration and exploitation
        pop_size = min(40, max(20, budget // 5))
        remaining = budget

        # Initial random population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        remaining -= pop_size

        # Non-dominated sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            front = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1
                if n_dom[i] == 0:
                    front[0].append(i)

            k = 0
            while front[k]:
                next_front = []
                for i in front[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                front.append(next_front)
            front.pop()  # remove empty last
            return front

        # Crowding distance
        def crowding_distance(F, front_idx):
            if len(front_idx) <= 2:
                return np.ones(len(front_idx))
            n = len(front_idx)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_idx, m]
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                if vals[sorted_idx[-1]] - vals[sorted_idx[0]] > 1e-12:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (vals[sorted_idx[i+1]] - vals[sorted_idx[i-1]]) / (vals[sorted_idx[-1]] - vals[sorted_idx[0]])
            return dist

        # Binary tournament selection
        def tournament_select(fronts, F, X):
            i = rng.integers(len(F))
            j = rng.integers(len(F))
            fi = next(k for k, fr in enumerate(fronts) if i in fr)
            fj = next(k for k, fr in enumerate(fronts) if j in fr)
            if fi < fj:
                return i
            elif fj < fi:
                return j
            else:
                # Same front, use crowding distance
                fr = fronts[fi]
                cd = crowding_distance(F, fr)
                pos_i = fr.index(i)
                pos_j = fr.index(j)
                return i if cd[pos_i] > cd[pos_j] else j

        # Crossover and mutation
        def crossover(x1, x2):
            if rng.random() < 0.9:
                alpha = rng.random()
                c1 = alpha * x1 + (1 - alpha) * x2
                c2 = (1 - alpha) * x1 + alpha * x2
                return c1, c2
            else:
                return x1.copy(), x2.copy()

        def mutate(x):
            x = x + 0.1 * (hi - lo) * rng.standard_normal(dim)
            return np.clip(x, lo, hi)

        # Main loop
        while remaining > 0:
            if time.time() > deadline:
                break

            fronts = fast_non_dominated_sort(F)

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, remaining)

            for _ in range(n_offspring):
                if time.time() > deadline:
                    break
                i = tournament_select(fronts, F, X)
                j = tournament_select(fronts, F, X)
                c1, c2 = crossover(X[i], X[j])
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                remaining -= 1
                if remaining <= 0:
                    break
                offspring_X.append(c2)
                offspring_F.append(problem(c2))
                remaining -= 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine and select next generation
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_fronts = fast_non_dominated_sort(combined_F)

            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to truncate this front by crowding distance
                    cd = crowding_distance(combined_F, front)
                    need = pop_size - len(next_X)
                    sorted_front = sorted(front, key=lambda idx: cd[front.index(idx)], reverse=True)
                    for idx in sorted_front[:need]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            if not next_X:
                break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
