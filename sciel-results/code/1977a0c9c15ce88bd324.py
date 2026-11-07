"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product


class Optimizer:
    """Multi-objective optimizer for RE37: LHS init + NSGA-II + weight-vector local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Use the global numpy RNG (seeded by evaluator via np.random.seed)
        # for diversity across seeds; wrap in a helper for consistent API.
        class _RNG:
            def __init__(self):
                pass
            def permutation(self, n):
                return np.random.permutation(n)
            def random(self, n=None):
                if n is None:
                    return np.random.random()
                return np.random.random(n)
            def integers(self, n):
                return np.random.randint(0, n)
            def standard_normal(self, size=None):
                return np.random.standard_normal(size)
        rng = _RNG()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Parameters
        pop_size = 30
        G = (self.budget - pop_size) // pop_size

        # 1. Latin Hypercube Sampling for initial population
        def lhs_sample(n, d, lo, hi, rng):
            samples = np.zeros((n, d))
            for dd in range(d):
                perm = rng.permutation(n)
                samples[:, dd] = (perm + rng.random(n)) / n
            return lo + (hi - lo) * samples

        X = lhs_sample(pop_size, dim, lo, hi, rng)
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
                u = rng.random()
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
                if rng.random() < 1.0 / dim:
                    sigma = sigma_scale * (hi[i] - lo[i])
                    child[i] += sigma * rng.standard_normal()
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
                    best_idx = rng.integers(pop_size)
                    for _ in range(k - 1):
                        cand = rng.integers(pop_size)
                        if ranks[cand] < ranks[best_idx]:
                            best_idx = cand
                        elif ranks[cand] == ranks[best_idx] and cd[cand] > cd[best_idx]:
                            best_idx = cand
                    return best_idx

                p1 = tournament_select(ranks, cd)
                p2 = tournament_select(ranks, cd)

                if rng.random() < 0.9:
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

        # 7. Weight-vector based local refinement around non-dominated points
        if used < self.budget and time.time() < deadline:
            ranks_final = fast_non_dominated_sort(F)
            nondom_mask = (ranks_final == 0)
            nondom_idx = np.where(nondom_mask)[0]

            remaining = self.budget - used

            # Generate weight vectors for scalarization
            # For 3 objectives, use simplex grid with more levels for finer coverage
            def generate_weights(n_obj, n_levels=5):
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

            weight_vectors = generate_weights(n_obj, n_levels=5)

            # For each weight vector, do a local search from the best non-dominated point
            # that aligns with that weight direction
            for w in weight_vectors:
                if used >= self.budget or time.time() > deadline:
                    break

                # Find the non-dominated point that best matches this weight direction
                # Minimize the scalarized objective
                if len(nondom_idx) > 0:
                    scalarized = F[nondom_idx] @ w
                    best_local_idx = nondom_idx[np.argmin(scalarized)]
                else:
                    continue

                base = X[best_local_idx].copy()
                scalar_base = F[best_local_idx] @ w

                # Local search: try small perturbations with decreasing step size
                for step in range(6):
                    if used >= self.budget or time.time() > deadline:
                        break

                    # Decrease step size
                    scale = 0.08 * (0.65 ** step) * (hi - lo)
                    for trial in range(2):
                        if used >= self.budget or time.time() > deadline:
                            break

                        perturbed = base + scale * rng.standard_normal(dim)
                        perturbed = np.clip(perturbed, lo, hi)
                        fp = problem(perturbed)
                        used += 1

                        # Check if perturbed improves the scalarized objective
                        scalar_new = fp @ w

                        if scalar_new < scalar_base:
                            # Accept if it's not dominated by any existing point
                            is_dom = False
                            for j in range(len(F)):
                                if np.all(F[j] <= fp) and np.any(F[j] < fp):
                                    is_dom = True
                                    break
                            if not is_dom:
                                # Replace the point
                                F[best_local_idx] = fp
                                X[best_local_idx] = perturbed
                                base = perturbed.copy()
                                scalar_base = scalar_new

        return X
# EVOLVE-BLOCK-END
