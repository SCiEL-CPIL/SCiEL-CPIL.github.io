import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        N = 20
        if self.budget < N:
            N = self.budget

        # Latin Hypercube Sampling for initial population
        X = np.zeros((N, self.dim))
        for d in range(self.dim):
            edges = np.linspace(0.0, 1.0, N + 1)
            for i in range(N):
                X[i, d] = np.random.uniform(edges[i], edges[i + 1])
        # Map to actual bounds
        X = lo + X * span

        F = np.array([problem(x) for x in X])
        used = N

        # DE parameters
        F_factor = 0.8
        CR = 0.5

        while used + N <= self.budget:
            X_trial = np.zeros_like(X)

            for i in range(N):
                # Select 3 distinct parents
                indices = list(range(N))
                indices.remove(i)
                r1, r2, r3 = np.random.choice(indices, size=3, replace=False)

                # DE/rand/1
                v = X[r1] + F_factor * (X[r2] - X[r3])

                # Binary Crossover
                for d in range(self.dim):
                    if np.random.rand() < CR:
                        X_trial[i, d] = v[d]
                    else:
                        X_trial[i, d] = X[i, d]

                # Ensure at least one gene is from v
                if np.all(X_trial[i] == X[i]):
                    d_rand = np.random.randint(self.dim)
                    X_trial[i, d_rand] = v[d_rand]

                X_trial[i] = np.clip(X_trial[i], lo, hi)

            F_trial = np.array([problem(x) for x in X_trial])
            used += N

            # Combine and Select using NSGA-II style
            X_all = np.vstack([X, X_trial])
            F_all = np.vstack([F, F_trial])

            ranks, fronts = self._fast_non_dominated_sort(F_all)

            new_X = []
            new_F = []
            current_size = 0
            for i in range(len(fronts)):
                front = fronts[i]
                if current_size + len(front) <= N:
                    for idx in front:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                    current_size += len(front)
                else:
                    # Calculate crowding for this front
                    crowd_front = np.zeros(len(front))
                    for k in range(self.n_obj):
                        sorted_idx = np.argsort(F_all[front, k])
                        crowd_front[sorted_idx[0]] = np.inf
                        crowd_front[sorted_idx[-1]] = np.inf
                        min_val = F_all[front, k].min()
                        max_val = F_all[front, k].max()
                        range_val = max_val - min_val
                        if range_val == 0:
                            continue
                        for j in range(1, len(front) - 1):
                            idx = front[sorted_idx[j]]
                            idx_prev = front[sorted_idx[j - 1]]
                            idx_next = front[sorted_idx[j + 1]]
                            crowd_front[sorted_idx[j]] += (F_all[idx_next, k] - F_all[idx_prev, k]) / range_val

                    sorted_front_idx = np.argsort(crowd_front)[::-1]
                    needed = N - current_size
                    for j in range(needed):
                        idx = front[sorted_front_idx[j]]
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                    current_size += needed
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        S = [[] for _ in range(n)]
        n_d = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_d[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_d[i] += 1

        current_front = []
        for i in range(n):
            if n_d[i] == 0:
                ranks[i] = 0
                current_front.append(i)

        k = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            current_front = next_front
            k += 1

        return ranks, fronts
