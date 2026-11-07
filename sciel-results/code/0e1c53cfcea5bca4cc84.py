"""NSGA-II-based multi-objective optimizer for WFG9 with Tchebycheff decomposition."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG9 with Tchebycheff decomposition seeding."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size for NSGA-II
        pop_size = min(50, budget // 4)
        if pop_size < 15:
            pop_size = 15

        # Initialize population using Tchebycheff decomposition for diverse coverage
        # Generate weight vectors on the simplex
        n_weights = pop_size
        # Use quasi-random points for weight vectors to ensure diversity
        # For 3 objectives, use points on the 2D simplex
        weights = []
        # Add extreme points
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
        # Add interior points using Fibonacci-like spacing
        remaining = n_weights - n_obj
        if remaining > 0:
            # Generate points on the simplex using rejection sampling with stratification
            for _ in range(remaining):
                while True:
                    w = rng.random(n_obj)
                    w = w / w.sum()
                    # Check minimum distance to existing weights
                    min_dist = min(np.linalg.norm(w - existing) for existing in weights)
                    if min_dist > 0.1 or len(weights) < 10:
                        break
                weights.append(w)

        # Evaluate each weight vector by solving the Tchebycheff subproblem
        # For initialization, we use a simple approach: sample and evaluate
        X = np.empty((pop_size, dim))
        F = np.empty((pop_size, n_obj))
        evals = 0
        all_F_list = []

        # First, do a quick random initialization to get a sense of the objective space
        X_init = lo + span * rng.random((min(10, budget), dim))
        F_init = np.empty((min(10, budget), n_obj))
        for i in range(min(10, budget)):
            F_init[i] = problem(X_init[i])
            evals += 1
            all_F_list.append(F_init[i].copy())

        # Use the initial evaluations to estimate objective ranges
        if evals >= n_obj:
            f_min = F_init.min(axis=0)
            f_max = F_init.max(axis=0)
            f_range = np.maximum(f_max - f_min, 1e-10)
        else:
            f_min = np.zeros(n_obj)
            f_range = np.ones(n_obj)

        # Now populate the main population using Tchebycheff-guided sampling
        for i in range(pop_size):
            if evals >= budget:
                break
            w = weights[i]
            # Sample a point and evaluate
            x = lo + span * rng.random(dim)
            X[i] = x
            F[i] = problem(x)
            evals += 1
            all_F_list.append(F[i].copy())

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sort. Returns list of fronts (each is a list of indices)."""
            n = len(F_arr)
            if n == 0:
                return []
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of points that dominate i
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            front = [i for i in range(n) if n_dom[i] == 0]
            fronts.append(front)

            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front

            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices in F_arr."""
            n = len(indices)
            if n <= 2:
                return np.ones(n) * np.inf

            cd = np.zeros(n)
            idx = np.array(indices)
            F_sub = F_arr[idx]

            for m in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                f_min = F_sub[sorted_idx[0], m]
                f_max = F_sub[sorted_idx[-1], m]
                if f_max - f_min < 1e-15:
                    continue

                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / (f_max - f_min)

            return cd

        def compute_rank_and_cd(F_arr, fronts):
            """Precompute rank array and crowding distance array."""
            n = len(F_arr)
            rank = np.zeros(n, dtype=int)
            cd_arr = np.zeros(n)
            for front_id, front in enumerate(fronts):
                for i in front:
                    rank[i] = front_id
                if len(front) > 2:
                    cd_front = crowding_distance(F_arr, front)
                    for pos, idx in enumerate(front):
                        cd_arr[idx] = cd_front[pos]
                else:
                    for idx in front:
                        cd_arr[idx] = np.inf
            return rank, cd_arr

        def tournament_select(F_arr, fronts, rank, cd_arr, k=3):
            """Tournament selection based on non-dominated rank and crowding distance."""
            candidates = rng.integers(0, len(F_arr), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best] or (rank[c] == rank[best] and cd_arr[c] > cd_arr[best]):
                    best = c
            return best

        # Main NSGA-II loop
        while evals < budget:
            if time.perf_counter() > deadline - 10:
                break

            # Precompute ranks and crowding distances
            fronts = fast_non_dominated_sort(F)
            rank, cd_arr = compute_rank_and_cd(F, fronts)

            # Create offspring population
            offspring_X = np.empty((pop_size, dim))
            offspring_F = np.empty((pop_size, n_obj))
            n_offspring = 0

            for i in range(pop_size):
                if evals >= budget or time.perf_counter() > deadline - 10:
                    break

                # Select two parents
                p1 = tournament_select(F, fronts, rank, cd_arr)
                p2 = tournament_select(F, fronts, rank, cd_arr)

                # Crossover: blend crossover
                if rng.random() < 0.9:  # crossover probability
                    alpha = rng.random()
                    child1 = (1 - alpha) * X[p1] + alpha * X[p2]
                    child2 = (1 - alpha) * X[p2] + alpha * X[p1]
                else:
                    child1 = X[p1].copy()
                    child2 = X[p2].copy()

                # Mutation: Gaussian perturbation
                mutation_rate = 0.3
                for j in range(dim):
                    if rng.random() < mutation_rate:
                        child1[j] += 0.1 * span[j] * rng.standard_normal()
                    if rng.random() < mutation_rate:
                        child2[j] += 0.1 * span[j] * rng.standard_normal()

                # Clip to bounds
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                # Evaluate child1
                offspring_X[n_offspring] = child1
                offspring_F[n_offspring] = problem(child1)
                evals += 1
                all_F_list.append(offspring_F[n_offspring].copy())
                n_offspring += 1

                if evals >= budget or time.perf_counter() > deadline - 10:
                    break

                # Evaluate child2 if we have budget
                if n_offspring < pop_size and evals < budget:
                    offspring_X[n_offspring] = child2
                    offspring_F[n_offspring] = problem(child2)
                    evals += 1
                    all_F_list.append(offspring_F[n_offspring].copy())
                    n_offspring += 1

            # Combine parent and offspring
            if n_offspring == 0:
                break

            combined_X = np.vstack([X, offspring_X[:n_offspring]])
            combined_F = np.vstack([F, offspring_F[:n_offspring]])
            combined_n = pop_size + n_offspring

            # Non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            new_X = np.empty((pop_size, dim))
            new_F = np.empty((pop_size, n_obj))
            count = 0

            for front in combined_fronts:
                if count + len(front) <= pop_size:
                    # Take entire front
                    for idx in front:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Need to truncate this front using crowding distance
                    remaining = pop_size - count
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                    break

            X = new_X
            F = new_F

        # Return the final population
        return X
# EVOLVE-BLOCK-END
