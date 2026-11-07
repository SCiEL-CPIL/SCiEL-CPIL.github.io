# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II for DTLZ4 exploiting structure: x[3:] kept tiny, focus on x[0:3]."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4: x[0:3] shape the front, x[3:] affect g with alpha=100
        # Strategy: keep x[3:] very small, optimize x[0:3]
        n_hard = 3  # x[0:3]
        n_easy = dim - n_hard  # x[3:]

        def make_x(rng):
            # x[0:3]: uniform in [0,1]
            x_hard = rng.random(n_hard)
            # x[3:]: very small values to keep g near 0
            x_easy = rng.random(n_easy) * 0.005
            x = np.zeros(dim)
            x[:n_hard] = x_hard
            x[n_hard:] = x_easy
            return x

        # Population size
        pop_size = min(20, budget // 2)
        if pop_size < 5:
            pop_size = 5

        # Initialize population
        X = np.array([make_x(rng) for _ in range(pop_size)])
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts.append([i])
            k = 0
            while k < len(fronts):
                Q = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            Q.append(j)
                if len(Q) == 0:
                    break
                fronts.append(Q)
                k += 1
            return fronts

        def crowding_distance(F, idx):
            n = len(idx)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            F_sub = F[idx]
            for m in range(n_obj):
                vals = F_sub[:, m]
                vmin, vmax = vals.min(), vals.max()
                if vmax - vmin < 1e-15:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (vmax - vmin)
            return dist

        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring
            X_off = np.array([make_x(rng) for _ in range(pop_size)])
            F_off = np.array([problem(x) for x in X_off])
            used += pop_size

            # Combine parent and offspring
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])

            # Non-dominated sort
            fronts = non_dominated_sort(F_all)

            # Select next population
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    X_new.append(X_all[front])
                    F_new.append(F_all[front])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(X_new)
                    cd = crowding_distance(F_all, front)
                    order = np.argsort(-cd)
                    selected = [front[i] for i in order[:remaining]]
                    X_new.append(X_all[selected])
                    F_new.append(F_all[selected])
                    break

            X = np.vstack(X_new)
            F = np.vstack(F_new)

            # Apply mutation to next generation's parents (for diversity)
            # Actually, we already created fresh offspring. Let's add some mutation.
            # For the next iteration, we'll mutate some parents
            for i in range(pop_size):
                if rng.random() < 0.8:
                    # Mutate x[0:3] with Gaussian, keep x[3:] tiny
                    X[i, :n_hard] += 0.05 * rng.standard_normal(n_hard)
                    X[i, :n_hard] = np.clip(X[i, :n_hard], 0, 1)
                    # Re-sample x[3:] to keep them small
                    X[i, n_hard:] = rng.random(n_easy) * 0.005

        return X
# EVOLVE-BLOCK-END
