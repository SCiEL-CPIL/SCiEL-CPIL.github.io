"""Multi-objective optimizer on WFG7 using Sobol initialization + NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """NSGA-II with Sobol lattice initialization for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: use ~20 for good coverage
        pop_size = 20
        # Crossover probability
        pc = 0.9
        # Mutation probability per gene
        pm = 1.0 / self.dim
        # Sigma for Gaussian mutation
        sigma = 0.15

        # Sobol lattice initialization for uniform coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Generate points: use power of 2 for Sobol
        n_init = pop_size
        # Sobol requires power of 2, so use 32 and take first 20
        n_sobol = 1
        while n_sobol < n_init:
            n_sobol *= 2
        sobol_samples = sampler.random(n_sobol)
        X = lo + span * sobol_samples[:n_init]
        F = np.array([problem(x) for x in X])
        used = n_init

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

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Sort into fronts
            fronts = fast_non_dominated_sort(F)

            # Precompute rank for each individual
            rank = np.full(pop_size, pop_size, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Precompute crowding distances for each front
            cd_by_front = {}
            for r, front in enumerate(fronts):
                cd_by_front[r] = crowding_distance(F, front)

            def tournament():
                i = np.random.randint(pop_size)
                j = np.random.randint(pop_size)
                if rank[i] < rank[j]:
                    return i
                elif rank[j] < rank[i]:
                    return j
                else:
                    # Same front, use crowding distance
                    r = rank[i]
                    cd = cd_by_front[r]
                    idx_i = front.index(i) if i in front else 0
                    idx_j = front.index(j) if j in front else 0
                    # Find index within the front
                    fr = fronts[r]
                    idx_i = fr.index(i)
                    idx_j = fr.index(j)
                    return i if cd[idx_i] > cd[idx_j] else j

            # Generate offspring
            children = []
            child_F = []
            while len(children) < pop_size and used < self.budget:
                p1 = tournament()
                p2 = tournament()
                if np.random.random() < pc:
                    # Simulated binary crossover
                    u = np.random.random()
                    beta = 1.0 / (2.0 * u) if u <= 0.5 else 1.0 / (2.0 * (1.0 - u))
                    beta = beta ** (1.0 / (2.0 + 1.0))
                    c1 = 0.5 * ((1 + beta) * X[p1] + (1 - beta) * X[p2])
                    c2 = 0.5 * ((1 - beta) * X[p1] + (1 + beta) * X[p2])
                else:
                    c1 = X[p1].copy()
                    c2 = X[p2].copy()

                # Gaussian mutation
                for c in (c1, c2):
                    mask = np.random.random(self.dim) < pm
                    c[mask] += sigma * span[mask] * np.random.standard_normal(mask.sum())
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
