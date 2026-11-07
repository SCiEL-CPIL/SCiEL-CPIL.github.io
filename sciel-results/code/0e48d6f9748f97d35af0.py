"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin: evaluator kills at 330s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.RandomState(42)

        N = 30  # population size
        used = 0
        records = []  # list of objective vectors

        def evaluate(x):
            nonlocal used
            used += 1
            f = np.array(problem(x), dtype=float)
            records.append(f)
            return f

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_count = [0] * n
            fronts = [[]]
            
            # Vectorized dominance check
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_count[i] += 1
                    # else mutually non-dominated

            for i in range(n):
                if n_count[i] == 0:
                    fronts[0].append(i)
            
            k = 1
            while fronts[k - 1]:
                current = fronts[k - 1]
                next_front = []
                for i in current:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(self.n_obj):
                # Sort indices by objective m
                sorted_indices = sorted(range(n), key=lambda i: F[front_indices[i]][m])
                # Min and max get inf
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F[front_indices[sorted_indices[0]]][m]
                f_max = F[front_indices[sorted_indices[-1]]][m]
                if f_max == f_min:
                    continue
                    
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    dist[idx] += (F[front_indices[next_idx]][m] - F[front_indices[prev_idx]][m]) / (f_max - f_min)
            return dist

        def tournament_select(pop_X, pop_F, ranks, cds):
            # ranks: list of front indices for each individual
            # cds: list of crowding distances for each individual
            i = rng.randint(len(pop_X))
            j = rng.randint(len(pop_X))
            
            if ranks[i] < ranks[j]:
                return pop_X[i]
            elif ranks[j] < ranks[i]:
                return pop_X[j]
            else:
                if cds[i] > cds[j]:
                    return pop_X[i]
                else:
                    return pop_X[j]

        def crossover(x1, x2):
            if rng.random() < 0.9:
                mask = rng.random(self.dim) < 0.5
                return np.where(mask, x1, x2)
            return x1.copy()

        def mutate(x, sigma):
            x = x + sigma * (hi - lo) * rng.standard_normal(self.dim)
            return np.clip(x, lo, hi)

        # Initial population
        pop_X = np.array([lo + (hi - lo) * rng.random(self.dim) for _ in range(N)])
        pop_F = np.array([evaluate(x) for x in pop_X])

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Sort current population
            fronts = fast_non_dominated_sort(pop_F)
            
            # Assign ranks and crowding distances
            ranks = [0] * N
            cds = [0.0] * N
            for rank_idx, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank_idx
                if len(front) > 1:
                    front_cds = crowding_distance(pop_F, front)
                    for i, idx in enumerate(front):
                        cds[idx] = front_cds[i]
                else:
                    if len(front) == 1:
                        cds[front[0]] = float('inf')

            # Create children
            children_X = []
            children_F = []
            for _ in range(N):
                p1 = tournament_select(pop_X, pop_F, ranks, cds)
                p2 = tournament_select(pop_X, pop_F, ranks, cds)
                child = crossover(p1, p2)
                sigma = 0.15
                child = mutate(child, sigma)
                fc = evaluate(child)
                children_X.append(child)
                children_F.append(fc)
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

            # Combine
            all_X = np.vstack([pop_X, np.array(children_X)])
            all_F = np.vstack([pop_F, np.array(children_F)])

            # Sort combined population
            all_fronts = fast_non_dominated_sort(all_F)
            total_N = len(all_X)
            
            # Assign ranks and crowding distances for combined
            all_ranks = [0] * total_N
            all_cds = [0.0] * total_N
            for rank_idx, front in enumerate(all_fronts):
                for idx in front:
                    all_ranks[idx] = rank_idx
                if len(front) > 1:
                    front_cds = crowding_distance(all_F, front)
                    for i, idx in enumerate(front):
                        all_cds[idx] = front_cds[i]
                else:
                    if len(front) == 1:
                        all_cds[front[0]] = float('inf')

            # Select next population
            new_X = []
            new_F = []
            for front in all_fronts:
                if len(new_X) + len(front) <= N:
                    for idx in front:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                else:
                    # Need crowding distance
                    front_cds = crowding_distance(all_F, front)
                    sorted_front = sorted(front, key=lambda i: front_cds[front.index(i)], reverse=True)
                    remaining = N - len(new_X)
                    for idx in sorted_front[:remaining]:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                    break

            pop_X = np.array(new_X)
            pop_F = np.array(new_F)

        return pop_X

# EVOLVE-BLOCK-END
