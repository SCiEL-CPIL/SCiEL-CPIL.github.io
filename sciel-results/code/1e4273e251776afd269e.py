"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        pop_size = min(60, max(20, self.budget // 4))
        # Crossover probability
        pc = 0.9
        # Mutation probability per gene
        pm = 1.0 / self.dim
        # Sigma for Gaussian mutation
        sigma = 0.2

        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        start_time = time.time()
        deadline = start_time + 310  # safety margin before 330s kill

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [set() for _ in range(n)]
            n_dominated = np.zeros(n, dtype=int)
            fronts = []
            current_front = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominated[i] += 1
                if n_dominated[i] == 0:
                    current_front.append(i)
            current_front = set(current_front)
            fronts.append(sorted(current_front))
            while current_front:
                next_front = set()
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.add(j)
                if not next_front:
                    break
                fronts.append(sorted(next_front))
                current_front = next_front
            return fronts

        def crowding_distance(F, front):
            """Assign crowding distance to indices in front."""
            n = len(front)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[front, m]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                val_range = vals[order[-1]] - vals[order[0]]
                if val_range > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
            return cd

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Sort into fronts
            fronts = fast_non_dominated_sort(F)

            # Selection: binary tournament
            def tournament():
                i = rng.integers(pop_size)
                j = rng.integers(pop_size)
                # Find front ranks
                rank_i = rank_j = pop_size
                for r, front in enumerate(fronts):
                    if i in front:
                        rank_i = r
                    if j in front:
                        rank_j = r
                    if rank_i < pop_size and rank_j < pop_size:
                        break
                if rank_i < rank_j:
                    return i
                elif rank_j < rank_i:
                    return j
                else:
                    # Same front, use crowding distance
                    front_idx = [r for r, fr in enumerate(fronts) if i in fr][0]
                    fr = fronts[front_idx]
                    cd = crowding_distance(F, fr)
                    idx_i = fr.index(i)
                    idx_j = fr.index(j)
                    return i if cd[idx_i] > cd[idx_j] else j

            # Generate offspring
            children = []
            child_F = []
            while len(children) < pop_size and used < self.budget:
                p1 = tournament()
                p2 = tournament()
                if rng.random() < pc:
                    # Simulated binary crossover
                    u = rng.random()
                    beta = 1.0 / (2.0 * u) if u <= 0.5 else 1.0 / (2.0 * (1.0 - u))
                    beta = beta ** (1.0 / (2.0 + 1.0))
                    c1 = 0.5 * ((1 + beta) * X[p1] + (1 - beta) * X[p2])
                    c2 = 0.5 * ((1 - beta) * X[p1] + (1 + beta) * X[p2])
                else:
                    c1 = X[p1].copy()
                    c2 = X[p2].copy()

                # Gaussian mutation
                for c in (c1, c2):
                    mask = rng.random(self.dim) < pm
                    c[mask] += sigma * span[mask] * rng.standard_normal(mask.sum())
                    np.clip(c, lo, hi, out=c)

                fc1 = problem(c1)
                used += 1
                fc2 = problem(c2)
                used += 1
                children.append(c1)
                child_F.append(fc1)
                children.append(c2)
                child_F.append(fc2)

            if not children:
                break

            children = np.array(children[:pop_size])
            child_F = np.array(child_F[:pop_size])

            # Combine parent and offspring
            all_X = np.vstack([X, children])
            all_F = np.vstack([F, child_F])

            # Sort combined population
            all_fronts = fast_non_dominated_sort(all_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in all_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(all_F, front)
                    order = np.argsort(-cd)
                    for i in order[:remaining]:
                        idx = front[i]
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                    break

            X = np.array(new_X[:pop_size])
            F = np.array(new_F[:pop_size])

        return X
# EVOLVE-BLOCK-END
