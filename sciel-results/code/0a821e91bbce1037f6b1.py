import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Population size for NSGA-II
        N = 20
        # Number of generations: N initial + N per generation
        G = (budget - N) // N

        # Initialize population with random points
        X = lo + (hi - lo) * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])

        # Main NSGA-II loop
        for gen in range(G):
            if not time_ok():
                break

            # Create offspring population Q of size N
            Q = np.zeros((N, dim))
            for i in range(0, N, 2):
                # Tournament selection
                p1 = self._tournament_select(X, F)
                p2 = self._tournament_select(X, F)

                # SBX Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)

                # Polynomial Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)

                Q[i] = c1
                if i + 1 < N:
                    Q[i + 1] = c2

            # Evaluate offspring
            FQ = np.array([problem(x) for x in Q])

            if not time_ok():
                break

            # Combine parents and offspring
            R_X = np.vstack((X, Q))
            R_F = np.vstack((F, FQ))

            # Non-dominated sorting
            fronts = self._non_dominated_sort(R_F)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= N:
                    next_X.append(R_X[front])
                    next_F.append(R_F[front])
                else:
                    remaining = N - len(next_X)
                    subset_F = R_F[front]
                    cd = self._crowding_distance(subset_F)
                    sorted_indices = np.argsort(-cd)
                    selected_indices = sorted_indices[:remaining]
                    next_X.append(R_X[front][selected_indices])
                    next_F.append(R_F[front][selected_indices])
                    break

            X = np.vstack(next_X)
            F = np.vstack(next_F)

        return X

    def _tournament_select(self, X, F):
        """Tournament selection of size 2 based on dominance."""
        n = len(X)
        idx1 = np.random.randint(n)
        idx2 = np.random.randint(n)
        
        f1 = F[idx1]
        f2 = F[idx2]
        
        # Check dominance
        if np.all(f1 <= f2) and np.any(f1 < f2):
            return X[idx1]
        elif np.all(f2 <= f1) and np.any(f2 < f1):
            return X[idx2]
        else:
            # If incomparable, use crowding distance as tie-breaker
            # For simplicity in tournament of 2, just pick randomly
            # Or we could compute a simple rank. Let's stick to random for speed
            # unless we precompute ranks.
            return X[np.random.choice([idx1, idx2])]

    def _sbx_crossover(self, p1, p2, lo, hi):
        """Simulated Binary Crossover."""
        eta_c = 20.0
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)

        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    x1, x2 = sorted([p1[i], p2[i]])
                    if np.random.rand() <= 0.5:
                        beta = (np.random.rand() ** (1.0 / (eta_c + 1)))
                        c1[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                        c2[i] = 0.5 * ((1 + 1/beta) * x1 + (1 - 1/beta) * x2)
                    else:
                        beta = (np.random.rand() ** (1.0 / (eta_c + 1)))
                        c1[i] = 0.5 * ((1 + beta) * x2 + (1 - beta) * x1)
                        c2[i] = 0.5 * ((1 + 1/beta) * x2 + (1 - 1/beta) * x1)
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]

        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)

        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        """Polynomial Mutation."""
        eta_m = 20.0
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:  # Mutation probability
                y = x[i]
                delta1 = (y - lo[i]) / (hi[i] - lo[i]) if (hi[i] - lo[i]) > 0 else 0
                delta2 = (hi[i] - y) / (hi[i] - lo[i]) if (hi[i] - lo[i]) > 0 else 0
                mut_pow = 1.0 / (eta_m + 1)

                u = np.random.rand()
                if u < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy) ** (mut_pow + 1)
                    deltaq = val ** (1.0 / (mut_pow + 1)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy) ** (mut_pow + 1)
                    deltaq = val ** (1.0 / (mut_pow + 1)) - 1.0

                y = y + deltaq * (hi[i] - lo[i])
                x[i] = np.clip(y, lo[i], hi[i])
        return x

    def _non_dominated_sort(self, F):
        """Non-dominated sorting. Returns list of fronts (each is array of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        n_d = np.zeros(n, dtype=int)
        rank = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_d[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_d[i] += 1

        fronts = []
        current_front = np.where(n_d == 0)[0]
        rank[current_front] = 0

        while len(current_front) > 0:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_d[j] -= 1
                    if n_d[j] == 0:
                        rank[j] = rank[i] + 1
                        next_front.append(j)
            current_front = np.array(next_front)

        return fronts

    def _crowding_distance(self, F):
        """Calculate crowding distance for a front."""
        m = len(F)
        if m <= 2:
            return np.full(m, np.inf)

        cd = np.zeros(m)
        for k in range(self.n_obj):
            sorted_indices = np.argsort(F[:, k])
            cd[sorted_indices[0]] = np.inf
            cd[sorted_indices[-1]] = np.inf
            f_min = F[sorted_indices[0], k]
            f_max = F[sorted_indices[-1], k]
            if f_max - f_min > 1e-14:
                for i in range(1, m - 1):
                    cd[sorted_indices[i]] += (F[sorted_indices[i+1], k] - F[sorted_indices[i-1], k]) / (f_max - f_min)
        return cd
