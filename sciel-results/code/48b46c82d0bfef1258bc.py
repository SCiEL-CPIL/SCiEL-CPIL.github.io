"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT4 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # ZDT4 specific: x0 in [0,1], x1..x9 in [-5,5]
        # f1 = x0
        # f2 = (1+sqrt(f1)) * g(x)
        # g(x) = sum((xi-2)^2) for i>=1
        # Pareto front: x0 in [0,1], g=0 => xi=2 for all i>=1
        # So f1 in [0,1], f2 = (1+sqrt(f1)) * 0 = 0 at optimum
        # But we need to explore the trade-off

        # Initialize population
        pop_size = min(100, max(20, budget // 2))
        pop_size = min(pop_size, budget)

        # Generate initial population
        X = np.zeros((pop_size, dim))
        # x0: uniform in [0,1]
        X[:, 0] = lo[0] + (hi[0] - lo[0]) * np.random.rand(pop_size)
        # x1..x9: mix of random and near-optimal (xi=2)
        for j in range(1, dim):
            # 50% chance to be near 2, 50% random
            near_opt = np.random.rand(pop_size) < 0.5
            X[near_opt, j] = 2.0 + 0.5 * np.random.randn(pop_size)
            X[~near_opt, j] = lo[j] + (hi[j] - lo[j]) * np.random.rand(pop_size[~near_opt])
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # NSGA-II helpers
        def fast_non_dominated_sort(F):
            """Returns list of fronts, each containing indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination count
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominated by i
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    front0.append(i)
            fronts.append(front0)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            F_sub = F[front_indices]
            for m in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / f_range
            return dist

        def tournament_select(F, fronts, pop_size, n):
            """Select n individuals via tournament selection."""
            selected = []
            for _ in range(n):
                candidates = np.random.randint(0, pop_size, size=2)
                # Compare by front rank then crowding
                f1 = [i for i in range(len(fronts)) if candidates[0] in fronts[i]]
                f2 = [i for i in range(len(fronts)) if candidates[1] in fronts[i]]
                rank1 = f1[0] if f1 else len(fronts)
                rank2 = f2[0] if f2 else len(fronts)
                if rank1 < rank2:
                    selected.append(candidates[0])
                elif rank2 < rank1:
                    selected.append(candidates[1])
                else:
                    # Same front, use crowding distance
                    front_idx = rank1
                    if front_idx < len(fronts):
                        front = fronts[front_idx]
                        cd = crowding_distance(F, front)
                        pos1 = np.where(np.array(front) == candidates[0])[0]
                        pos2 = np.where(np.array(front) == candidates[1])[0]
                        if len(pos1) > 0 and len(pos2) > 0:
                            if cd[pos1[0]] >= cd[pos2[0]]:
                                selected.append(candidates[0])
                            else:
                                selected.append(candidates[1])
                        else:
                            selected.append(candidates[0])
                    else:
                        selected.append(candidates[0])
            return selected

        def sbx_crossover(x1, x2, lo, hi, prob=0.9, eta=20.0):
            """Simulated binary crossover."""
            if np.random.rand() > prob:
                return x1.copy(), x2.copy()
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(len(x1)):
                if np.random.rand() < 0.5:
                    if x1[i] != x2[i]:
                        if x1[i] < x2[i]:
                            x_l, x_u = x1[i], x2[i]
                            c1, c2 = child1[i], child2[i]
                        else:
                            x_l, x_u = x2[i], x1[i]
                            c1, c2 = child2[i], child1[i]
                        beta = 1.0 + 2.0 * (x_l - lo[i]) / (x_u - x_l + 1e-30)
                        if np.random.rand() <= 1.0 / (eta + 1.0):
                            alpha = 2.0 - beta ** (-(eta + 1.0))
                        else:
                            alpha = (0.5 * (beta ** (eta + 1.0) - (2.0 - beta ** (eta + 1.0)))) ** (-1.0 / (eta + 1.0))
                        c1 = 0.5 * ((x_l + x_u) - alpha * (x_u - x_l))
                        if np.random.rand() <= 0.5:
                            child1[i] = c1
                        else:
                            child2[i] = c1
                        c2 = 0.5 * ((x_l + x_u) + alpha * (x_u - x_l))
                        if np.random.rand() <= 0.5:
                            child1[i] = c2
                        else:
                            child2[i] = c2
                    else:
                        pass
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def polynomial_mutation(x, lo, hi, prob=1.0 / dim, eta=20.0):
            """Polynomial mutation."""
            x_new = x.copy()
            for i in range(len(x)):
                if np.random.rand() < prob:
                    y = x_new[i]
                    if y <= lo[i] or y >= hi[i]:
                        y = (lo[i] + hi[i]) / 2.0
                    delta1 = (y - lo[i]) / (hi[i] - lo[i] + 1e-30)
                    delta2 = (hi[i] - y) / (hi[i] - lo[i] + 1e-30)
                    r = np.random.rand()
                    if r <= 0.5:
                        xy = (2.0 * r + (1.0 - 2.0 * r) * (delta1 ** (eta + 1.0))) ** (1.0 / (eta + 1.0))
                        deltaq = (xy ** (eta + 1.0)) - 1.0
                    else:
                        xy = (2.0 * (1.0 - r) + 2.0 * (r - 0.5) * ((1.0 - delta2) ** (eta + 1.0))) ** (1.0 / (eta + 1.0))
                        deltaq = 1.0 - (xy ** (eta + 1.0))
                    x_new[i] = y + deltaq * (hi[i] - lo[i])
            return np.clip(x_new, lo, hi)

        # Main NSGA-II loop
        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring = []
            while len(offspring) < pop_size:
                if time.time() > deadline:
                    break
                parents = tournament_select(F, fronts, pop_size, 2)
                p1, p2 = X[parents[0]], X[parents[1]]
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                offspring.append(c1)
                offspring.append(c2)
                used += 2
                if used >= budget:
                    break

            if not offspring:
                break

            offspring = np.array(offspring)
            F_off = np.array([problem(x) for x in offspring], dtype=float)
            used += len(offspring)

            # Combine parent and offspring
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, F_off])

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F_all)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(front)
                    new_F.append(front)
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(new_X)
                    front_arr = np.array(front)
                    cd = crowding_distance(F_all, front)
                    sorted_by_cd = np.argsort(-cd)
                    selected_in_front = front_arr[sorted_by_cd[:remaining]]
                    new_X.append(selected_in_front)
                    new_F.append(selected_in_front)
                    break

            # Flatten
            all_indices = np.concatenate(new_X)
            X = X_all[all_indices]
            F = F_all[all_indices]

            # Re-compute fronts for next iteration
            fronts = fast_non_dominated_sort(F)

        return X
# EVOLVE-BLOCK-END
