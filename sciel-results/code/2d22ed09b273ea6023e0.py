"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = 50
        if budget < pop_size * 2:
            pop_size = max(10, budget // 2)

        # Initialize population
        pop = lo + (hi - lo) * np.random.random((pop_size, dim))
        pop_f = np.array([problem(x) for x in pop], dtype=float)
        evaluations = pop_size

        # Crossover and mutation parameters
        crossover_prob = 0.9
        mutation_prob = 0.1
        eta_c = 20.0  # distribution index for crossover
        eta_m = 20.0  # distribution index for mutation

        def fast_non_dominated_sort(F):
            """Perform fast non-dominated sort. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_d = np.zeros(n, dtype=int)  # number of points dominating i
            rank = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_d[i] += 1
                if n_d[i] == 0:
                    rank[i] = 0
                    fronts.append([])
                    fronts[-1].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for k in fronts[i]:
                    for m in S[k]:
                        n_d[m] -= 1
                        if n_d[m] == 0:
                            rank[m] = i + 1
                            next_front.append(m)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {i: np.inf for i in indices}

            distances = {i: 0.0 for i in indices}
            n_obj = F.shape[1]

            for m in range(n_obj):
                # Sort by objective m
                sorted_indices = sorted(indices, key=lambda idx: F[idx, m])
                distances[sorted_indices[0]] = np.inf
                distances[sorted_indices[-1]] = np.inf

                f_min = F[sorted_indices[0], m]
                f_max = F[sorted_indices[-1], m]
                if f_max - f_min < 1e-15:
                    continue

                for k in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[k]
                    distances[idx] += (F[sorted_indices[k + 1], m] - F[sorted_indices[k - 1], m]) / (f_max - f_min)

            return distances

        def binary_tournament_selection(pop, pop_f, fronts):
            """Select two parents using binary tournament on non-dominated rank and crowding distance."""
            # Compute crowding distances for each front
            front_crowding = []
            for front in fronts:
                cd = crowding_distance(pop_f, front)
                front_crowding.append(cd)

            def get_crowding(idx):
                for i, front in enumerate(fronts):
                    if idx in front:
                        return front_crowding[i][idx]
                return 0.0

            def compare(i, j):
                # Compare by rank first, then crowding distance
                rank_i = 0
                rank_j = 0
                for r, front in enumerate(fronts):
                    if i in front:
                        rank_i = r
                        break
                    if j in front:
                        rank_j = r
                        break
                if rank_i < rank_j:
                    return i
                if rank_j < rank_i:
                    return j
                # Same rank, compare crowding distance
                cd_i = get_crowding(i)
                cd_j = get_crowding(j)
                return i if cd_i > cd_j else j

            # Tournament for parent 1
            i1 = np.random.randint(0, len(pop))
            i2 = np.random.randint(0, len(pop))
            p1 = compare(i1, i2)

            # Tournament for parent 2
            i1 = np.random.randint(0, len(pop))
            i2 = np.random.randint(0, len(pop))
            p2 = compare(i1, i2)

            return p1, p2

        def sbx_crossover(x1, x2):
            """Simulated Binary Crossover."""
            if np.random.random() > crossover_prob:
                return x1.copy(), x2.copy()

            child1 = x1.copy()
            child2 = x2.copy()

            for i in range(dim):
                if x1[i] == x2[i]:
                    continue
                if np.random.random() < 0.5:
                    if x1[i] > x2[i]:
                        x1[i], x2[i] = x2[i], x1[i]
                    if x1[i] < lo[i]:
                        x1[i] = lo[i]
                    if x2[i] > hi[i]:
                        x2[i] = hi[i]

                    if np.abs(x2[i] - x1[i]) < 1e-14:
                        continue

                    beta = 1.0 + (2.0 * (x1[i] - lo[i]) / (x2[i] - x1[i]))
                    alpha = 2.0 - beta ** (-1.0 - (eta_c + 1.0))

                    if np.random.random() <= 0.5 / alpha:
                        beta_q = beta ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - beta)) ** (1.0 / (eta_c + 1.0))

                    child1[i] = 0.5 * ((x1[i] + x2[i]) - beta_q * (x2[i] - x1[i]))

                    beta = 1.0 + (2.0 * (hi[i] - x2[i]) / (x2[i] - x1[i]))
                    alpha = 2.0 - beta ** (-1.0 - (eta_c + 1.0))

                    if np.random.random() <= 0.5 / alpha:
                        beta_q = beta ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - beta)) ** (1.0 / (eta_c + 1.0))

                    child2[i] = 0.5 * ((x1[i] + x2[i]) + beta_q * (x2[i] - x1[i]))

                child1[i] = np.clip(child1[i], lo[i], hi[i])
                child2[i] = np.clip(child2[i], lo[i], hi[i])

            return child1, child2

        def polynomial_mutation(x):
            """Polynomial mutation."""
            for i in range(dim):
                if np.random.random() < mutation_prob:
                    if x[i] <= lo[i]:
                        x[i] = lo[i]
                        continue
                    if x[i] >= hi[i]:
                        x[i] = hi[i]
                        continue

                    delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])

                    if np.random.random() < 0.5:
                        xy = 2.0 * np.random.random()
                        mut_pow = 1.0 / (eta_m + 1.0)
                        if xy <= 0.0:
                            val = (2.0 * xy + (1.0 - 2.0 * xy) * (delta1 ** (1.0 + mut_pow))) ** (1.0 / (1.0 + mut_pow))
                            deltaq = (val - 1.0) * (hi[i] - lo[i])
                        else:
                            val = (2.0 * (1.0 - xy) + 2.0 * (xy - 0.5) * (1.0 - delta2) ** (1.0 + mut_pow)) ** (1.0 / (1.0 + mut_pow))
                            deltaq = (val - 1.0) * (hi[i] - lo[i])
                    else:
                        xy = 2.0 * (1.0 - np.random.random())
                        mut_pow = 1.0 / (eta_m + 1.0)
                        if xy <= 0.0:
                            val = (2.0 * xy + (1.0 - 2.0 * xy) * (delta2 ** (1.0 + mut_pow))) ** (1.0 / (1.0 + mut_pow))
                            deltaq = (1.0 - val) * (hi[i] - lo[i])
                        else:
                            val = (2.0 * (1.0 - xy) + 2.0 * (xy - 0.5) * (1.0 - delta1) ** (1.0 + mut_pow)) ** (1.0 / (1.0 + mut_pow))
                            deltaq = (1.0 - val) * (hi[i] - lo[i])

                    x[i] += deltaq
                    x[i] = np.clip(x[i], lo[i], hi[i])

            return x

        # Main evolution loop
        while evaluations < budget:
            # Create offspring population
            offspring = []
            offspring_f = []

            while len(offspring) < pop_size and evaluations + len(offspring) < budget:
                p1, p2 = binary_tournament_selection(pop, pop_f, fast_non_dominated_sort(pop_f))
                c1, c2 = sbx_crossover(pop[p1], pop[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)

                if evaluations < budget:
                    f1 = problem(c1)
                    evaluations += 1
                    offspring.append(c1)
                    offspring_f.append(f1)

                if evaluations < budget and len(offspring) < pop_size:
                    f2 = problem(c2)
                    evaluations += 1
                    offspring.append(c2)
                    offspring_f.append(f2)

            if len(offspring) == 0:
                break

            # Combine parent and offspring
            combined = np.vstack([pop, np.array(offspring)])
            combined_f = np.vstack([pop_f, np.array(offspring_f)])

            # Non-dominated sort
            fronts = fast_non_dominated_sort(combined_f)

            # Select next generation
            new_pop = []
            new_pop_f = []
            for front in fronts:
                if len(new_pop) + len(front) <= pop_size:
                    new_pop.extend(front)
                    new_pop_f.extend(front)
                else:
                    remaining = pop_size - len(new_pop)
                    cd = crowding_distance(combined_f, front)
                    sorted_front = sorted(front, key=lambda idx: cd[idx], reverse=True)
                    new_pop.extend(sorted_front[:remaining])
                    new_pop_f.extend(sorted_front[:remaining])
                    break

            if len(new_pop) == 0:
                break

            pop = combined[new_pop]
            pop_f = combined_f[new_pop]

        return pop
# EVOLVE-BLOCK-END
