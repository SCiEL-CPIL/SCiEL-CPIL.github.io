# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a fast non-dominated sorting approach with
    crowding distance, adapted for the WFG9 problem."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        pop_size = min(50, max(10, self.budget // 5))
        n_obj = self.n_obj
        dim = self.dim

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of dominators
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            fronts.pop()  # remove last empty front
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices in F."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            obj_idx = list(range(n_obj))
            for m in obj_idx:
                vals = F[indices, m]
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                val_range = vals[order[-1]] - vals[order[0]]
                if val_range == 0:
                    continue
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / val_range
            return dist

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sort into fronts
            fronts = fast_non_dominated_sort(F)

            # Select parents from first few fronts
            parent_pool = []
            for front in fronts:
                parent_pool.extend(front)
                if len(parent_pool) >= pop_size // 2:
                    break

            if not parent_pool:
                parent_pool = list(range(pop_size))

            # Generate children
            n_children = min(pop_size, self.budget - used)
            children_X = []
            for _ in range(n_children):
                if time.time() > deadline:
                    break
                # Pick two parents (tournament-like: pick from parent_pool)
                p1 = parent_pool[rng.integers(len(parent_pool))]
                p2 = parent_pool[rng.integers(len(parent_pool))]

                # SBX-like crossover with some mutation
                alpha = 20.0
                child = np.zeros(dim)
                for d in range(dim):
                    if rng.random() < 0.5:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (alpha + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1.0 / (alpha + 1))
                        child[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (alpha + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1.0 / (alpha + 1))
                        child[d] = 0.5 * ((1 - beta) * X[p1, d] + (1 + beta) * X[p2, d])

                # Polynomial mutation
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u < 0.5:
                            delta_q = (2 * u) ** (1.0 / 20.0) - 1
                        else:
                            delta_q = 1 - (2 * (1 - u)) ** (1.0 / 20.0)
                        child[d] += delta_q * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                children_X.append(child)

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array([problem(x) for x in children_X], dtype=np.float64)
            used += len(children_X)

            # Combine parents and children, select next generation
            combined_X = np.vstack([X, children_X])
            combined_F = np.vstack([F, children_F])
            combined_fronts = fast_non_dominated_sort(combined_F)

            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.extend(front)
                    new_F.extend(front)
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(cd)[::-1]
                    selected = [front[i] for i in order[:remaining]]
                    new_X.extend(selected)
                    new_F.extend(selected)
                    break

            if not new_X:
                break

            X = combined_X[new_X]
            F = combined_F[new_F]

        return X
# EVOLVE-BLOCK-END
