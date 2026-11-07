"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE37: LHS init + NSGA-II + scalarized polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Use global numpy RNG (seeded by evaluator) for seed diversity
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Parameters
        pop_size = 40
        G = (self.budget - pop_size) // pop_size

        # 1. Latin Hypercube Sampling for initial population
        def lhs_sample(n, d, lo, hi):
            samples = np.zeros((n, d))
            for dd in range(d):
                perm = np.random.permutation(n)
                samples[:, dd] = (perm + np.random.random(n)) / n
            return lo + (hi - lo) * samples

        X = lhs_sample(pop_size, dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # 2. Fast non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            front = []

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[p] += 1
                if n_dom[p] == 0:
                    ranks[p] = 0
                    front.append(p)

            i = 0
            while len(front) > 0:
                i += 1
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            ranks[q] = i
                            next_front.append(q)
                front = next_front

            return ranks

        # 3. Crowding distance
        def crowding_distance(F, ranks):
            n = len(F)
            cd = np.zeros(n)
            unique_ranks = np.unique(ranks)

            for rank in unique_ranks:
                members = np.where(ranks == rank)[0]
                if len(members) <= 2:
                    cd[members] = np.inf
                    continue

                for m in range(n_obj):
                    obj_vals = F[members, m]
                    sorted_idx = np.argsort(obj_vals)
                    cd[members[sorted_idx[0]]] = np.inf
                    cd[members[sorted_idx[-1]]] = np.inf
                    obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                    if obj_range > 1e-12:
                        for k in range(1, len(sorted_idx) - 1):
                            cd[members[sorted_idx[k]]] += (
                                obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]
                            ) / obj_range

            return cd

        # 4. SBX crossover
        def sbx_crossover(x1, x2, beta=15):
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for i in range(dim):
                if x1[i] == x2[i]:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                    continue
                u = np.random.random()
                if u <= 0.5:
                    beta_c = (2 * u) ** (-1) if u > 0 else 1e6
                    beta_c = max(beta_c, 1e-6)
                    child1[i] = 0.5 * ((1 + beta_c) * x1[i] + (1 - beta_c) * x2[i])
                else:
                    beta_c = (2 * (1 - u)) ** (-1) if u < 1 else 1e6
                    beta_c = max(beta_c, 1e-6)
                    child1[i] = 0.5 * ((1 + beta_c) * x2[i] + (1 - beta_c) * x1[i])
                child2[i] = hi[i] + lo[i] - child1[i]
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        # 5. Mutation
        def mutate(child, sigma_scale=0.1):
            for i in range(dim):
                if np.random.random() < 1.0 / dim:
                    sigma = sigma_scale * (hi[i] - lo[i])
                    child[i] += sigma * np.random.standard_normal()
                    child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        # 6. Main NSGA-II evolution loop
        for gen in range(G):
            if time.time() > deadline:
                break

            ranks = fast_non_dominated_sort(F)
            cd = crowding_distance(F, ranks)

            offspring_X = []
            offspring_F = []

            for _ in range(pop_size // 2):
                # Tournament selection
                def tournament_select(ranks, cd, k=3):
                    best_idx = np.random.randint(pop_size)
                    for _ in range(k - 1):
                        cand = np.random.randint(pop_size)
                        if ranks[cand] < ranks[best_idx]:
                            best_idx = cand
                        elif ranks[cand] == ranks[best_idx] and cd[cand] > cd[best_idx]:
                            best_idx = cand
                    return best_idx

                p1 = tournament_select(ranks, cd)
                p2 = tournament_select(ranks, cd)

                if np.random.random() < 0.9:
                    c1, c2 = sbx_crossover(X[p1], X[p2])
                else:
                    c1, c2 = X[p1].copy(), X[p2].copy()

                c1 = mutate(c1)
                c2 = mutate(c2)

                f1 = problem(c1)
                used += 1
                f2 = problem(c2)
                used += 1

                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)

                if used >= self.budget:
                    break

                if time.time() > deadline:
                    break

            if used >= self.budget:
                break

            # Combine and select
            X_new = np.vstack([X, np.array(offspring_X)])
            F_new = np.vstack([F, np.array(offspring_F)])

            ranks_new = fast_non_dominated_sort(F_new)
            cd_new = crowding_distance(F_new, ranks_new)

            sort_idx = np.lexsort((-cd_new, ranks_new))
            X = X_new[sort_idx[:pop_size]]
            F = F_new[sort_idx[:pop_size]]

        # 7. Scalarized polish: use remaining budget to refine non-dominated points
        if used < self.budget and time.time() < deadline:
            ranks_final = fast_non_dominated_sort(F)
            nondom_mask = (ranks_final == 0)
            nondom_idx = np.where(nondom_mask)[0]

            remaining = self.budget - used

            # Generate weight vectors for scalarization
            def generate_weights(n_obj, n_levels=4):
                weights = []
                if n_obj == 3:
                    for i in range(n_levels + 1):
                        for j in range(n_levels + 1 - i):
                            k = n_levels - i - j
                            w = np.array([i, j, k], dtype=float)
                            if np.sum(w) > 0:
                                weights.append(w / np.sum(w))
                    return np.array(weights)
                else:
                    return np.array([np.ones(n_obj) / n_obj])

            weight_vectors = generate_weights(n_obj, n_levels=4)

            # For each weight vector, use scipy minimize to polish from best-matching point
            for w in weight_vectors:
                if used >= self.budget or time.time() > deadline:
                    break

                if len(nondom_idx) > 0:
                    scalarized = F[nondom_idx] @ w
                    best_local_idx = nondom_idx[np.argmin(scalarized)]
                else:
                    continue

                x0 = X[best_local_idx].copy()

                def scalarized_obj(x):
                    fx = problem(x)
                    used_local[0] += 1
                    return float(np.dot(fx, w))

                used_local = [0]
                try:
                    res = minimize(
                        scalarized_obj, x0,
                        method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                except Exception:
                    break

                if used_local[0] > 0:
                    used += used_local[0]
                    fx_res = problem(res.x)
                    # Check if this improves the scalarized objective and isn't dominated
                    scalar_new = float(np.dot(fx_res, w))
                    scalar_old = float(np.dot(F[best_local_idx], w))

                    if scalar_new < scalar_old:
                        is_dom = False
                        for j in range(len(F)):
                            if np.all(F[j] <= fx_res) and np.any(F[j] < fx_res):
                                is_dom = True
                                break
                        if not is_dom:
                            F[best_local_idx] = fx_res
                            X[best_local_idx] = res.x

        return X
# EVOLVE-BLOCK-END
