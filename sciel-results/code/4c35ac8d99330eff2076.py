"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective evolutionary algorithm for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        N = min(20, self.budget // 4)
        if N < 4:
            N = 4

        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        def fast_non_dominated_sort(F):
            """Returns list of fronts, each front is a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            nd = [0] * n  # number of solutions dominating i
            rank = [0] * n

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        nd[q] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        S[q].append(p)
                        nd[p] += 1

            fronts = []
            current_front = [i for i in range(n) if nd[i] == 0]
            for i in current_front:
                rank[i] = 0
            if current_front:
                fronts.append(current_front)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for q in S[i]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            rank[q] = k + 1
                            next_front.append(q)
                k += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices in F."""
            n = len(indices)
            if n <= 2:
                return np.ones(n)
            dist = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = np.argsort(F[indices, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F[indices[sorted_idx[-1]], m] - F[indices[sorted_idx[0]], m]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (
                        F[indices[sorted_idx[i + 1]], m] - F[indices[sorted_idx[i - 1]], m]
                    ) / f_range
            return dist

        def tournament_select(X, F, rank, k=2):
            """Tournament selection based on rank and crowding distance."""
            n = len(X)
            idx1 = rng.integers(n)
            idx2 = rng.integers(n)
            if rank[idx1] < rank[idx2]:
                return idx1
            elif rank[idx2] < rank[idx1]:
                return idx2
            # Same rank: use crowding distance
            fronts = fast_non_dominated_sort(F)
            # Find which front each is in
            front_map = {}
            for fi, front in enumerate(fronts):
                for idx in front:
                    front_map[idx] = fi
            cd = crowding_distance(F, list(range(n)))
            if cd[idx1] > cd[idx2]:
                return idx1
            else:
                return idx2

        def crossover(parent1, parent2):
            """Simulated binary crossover (SBX)."""
            if rng.random() < 0.5:
                return parent1.copy(), parent2.copy()
            eta = 20.0
            child1 = np.zeros_like(parent1)
            child2 = np.zeros_like(parent2)
            for i in range(dim):
                if parent1[i] != parent2[i]:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            return child1, child2

        def mutation(x):
            """Polynomial mutation."""
            eta = 20.0
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    # Scale relative to bounds
                    scale = (hi[i] - lo[i])
                    x_mut[i] += delta * scale
            return np.clip(x_mut, lo, hi)

        # Main loop
        while used < self.budget:
            # Compute ranks and crowding distances
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(len(X), dtype=int)
            for fi, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = fi

            # Generate offspring
            offspring_X = []
            offspring_F = []
            while len(offspring_X) < N and used + len(offspring_X) <= self.budget:
                p1 = tournament_select(X, F, rank)
                p2 = tournament_select(X, F, rank)
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutation(c1)
                c2 = mutation(c2)
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if used < self.budget and len(offspring_X) < N:
                    offspring_X.append(c2)
                    offspring_F.append(problem(c2))
                    used += 1

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])

            # Select next generation
            combined_fronts = fast_non_dominated_sort(combined_F)
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= N:
                    new_X.append(combined_X[front])
                    new_F.append(combined_F[front])
                else:
                    # Need to select some from this front based on crowding distance
                    need = N - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-cd)
                    selected = front[sorted_front[:need]]
                    new_X.append(combined_X[selected])
                    new_F.append(combined_F[selected])
                    break

            X = np.vstack(new_X)
            F = np.vstack(new_F)

        return X
# EVOLVE-BLOCK-END
