"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        B = self.budget

        # Population size
        pop_size = min(40, B // 5)
        if pop_size < 10:
            pop_size = 10

        # Initialize population with boundary emphasis for DTLZ4
        # DTLZ4: x_i^beta where beta=100, so most x_i should be 0 or 1
        X = np.empty((pop_size, n))
        for i in range(pop_size):
            x = np.zeros(n)
            # Last (n - n_obj + 1) variables are position variables
            # First (n_obj - 1) variables are angle variables
            n_pos = n - self.n_obj + 1  # 10 for dim=12, n_obj=3
            n_ang = self.n_obj - 1  # 2

            # Angle variables: uniform in [0, 1]
            x[:n_ang] = rng.random(n_ang)

            # Position variables: biased towards 0 or 1 (DTLZ4 characteristic)
            # Use a mixture: 50% near 0, 50% near 1
            for j in range(n_pos):
                if rng.random() < 0.5:
                    x[n_ang + j] = rng.beta(2, 20)  # concentrated near 0
                else:
                    x[n_ang + j] = 1 - rng.beta(2, 20)  # concentrated near 1

            X[i] = np.clip(x, lo, hi)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Non-dominated sorting
        def fast_non_dominated_sort(F):
            """Fast non-dominated sort, returns list of fronts (each is list of indices)."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            current_front = [i for i in range(n) if domination_count[i] == 0]
            fronts.append(current_front)

            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front

            return fronts

        # Crowding distance
        def crowding_distance(F, front_indices):
            """Calculate crowding distance for individuals in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for m in range(F.shape[1]):
                obj_vals = F[front_indices, m]
                min_val = obj_vals.min()
                max_val = obj_vals.max()
                if max_val - min_val < 1e-15:
                    continue
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]) / (max_val - min_val)
            return cd

        # Crowding distance assignment for the whole population
        def assign_crowding_distances(F, fronts):
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) == 0:
                    continue
                front_cd = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cd[idx] = front_cd[i]
            return cd

        # Selection: binary tournament
        def tournament_select(pop_indices, fronts_rank, cd, k=2):
            candidates = rng.choice(pop_indices, size=k, replace=False)
            best = candidates[0]
            for c in candidates[1:]:
                if fronts_rank[c] < fronts_rank[best]:
                    best = c
                elif fronts_rank[c] == fronts_rank[best] and cd[c] > cd[best]:
                    best = c
            return best

        # Crossover: SBX (Simulated Binary Crossover)
        def sbx_crossover(x1, x2, lo, hi, eta=20.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(len(x1)):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-15:
                        continue
                    if rng.random() < 0.5:
                        beta = 1 + 2 * (x1[i] - lo[i]) / (x2[i] - x1[i])
                    else:
                        beta = 1 + 2 * (hi[i] - x2[i]) / (x2[i] - x1[i])
                    if rng.random() <= 1.0 / (beta + 1):
                        alpha = (2 * beta ** (eta + 1) + (2 - beta ** (eta + 1))) ** (-1 / (eta + 1))
                    else:
                        alpha = (2 * (1 - beta) ** (eta + 1) + 2 * (beta - 1) ** (eta + 1)) ** (-1 / (eta + 1))
                    child1[i] = 0.5 * ((1 + alpha) * x1[i] + (1 - alpha) * x2[i])
                    child2[i] = 0.5 * ((1 - alpha) * x1[i] + (1 + alpha) * x2[i])
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        # Mutation: polynomial mutation
        def polynomial_mutate(x, lo, hi, eta=20.0, prob_mut=1.0 / n):
            x = x.copy()
            for i in range(len(x)):
                if rng.random() < prob_mut:
                    delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                    rand = rng.random()
                    if rand < 0.5:
                        xy = 1 - delta1
                        val = 2 * rand + (1 - 2 * rand) * (xy ** (eta + 1))
                        deltaq = (val ** (1 / (eta + 1)) - 1)
                    else:
                        xy = 1 - delta2
                        val = 2 * (1 - rand) + 2 * (rand - 0.5) * (xy ** (eta + 1))
                        deltaq = 1 - (val ** (1 / (eta + 1)))
                    x[i] += deltaq * (hi[i] - lo[i])
            return np.clip(x, lo, hi)

        # Main loop
        while used < B:
            # Evaluate if we have room
            if used + pop_size > B:
                # Evaluate one at a time
                fronts = fast_non_dominated_sort(F)
            else:
                fronts = fast_non_dominated_sort(F)

            # Assign ranks
            fronts_rank = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    fronts_rank[idx] = rank

            cd = assign_crowding_distances(F, fronts)

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size if used + pop_size <= B else (B - used)

            for _ in range(n_offspring):
                p1_idx = tournament_select(range(len(F)), fronts_rank, cd)
                p2_idx = tournament_select(range(len(F)), fronts_rank, cd)

                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx], lo, hi)
                c1 = polynomial_mutate(c1, lo, hi)
                c2 = polynomial_mutate(c2, lo, hi)

                offspring_X.append(c1)
                offspring_X.append(c2)

            # Evaluate offspring
            for off in offspring_X:
                if used >= B:
                    break
                f = problem(off)
                offspring_F.append(f)
                used += 1

            if not offspring_F:
                break

            offspring_F = np.array(offspring_F)
            offspring_X = np.array(offspring_X[:len(offspring_F)])

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Select next generation
            combined_fronts = fast_non_dominated_sort(combined_F)
            combined_cd = assign_crowding_distances(combined_F, combined_fronts)

            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Sort by crowding distance and take remaining
                    front_cd = combined_cd[front]
                    sorted_front = sorted(front, key=lambda i: front_cd[i], reverse=True)
                    remaining = pop_size - len(new_X)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
