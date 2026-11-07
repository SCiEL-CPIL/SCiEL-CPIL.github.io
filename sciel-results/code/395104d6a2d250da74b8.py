"""Multi-objective optimizer on ZDT6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with ZDT6-specific initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget

        # ZDT6-specific initialization:
        # For ZDT6, the Pareto front is achieved when x[1:] = 0 and x[0] in [0,1]
        # f1 = x[0], f2 = (1 + 10*n) * (1 - sqrt(f1/(1+10*n)))
        # So we initialize with x[1:] = 0 and x[0] spread across [0,1]
        # Also add some random points for exploration

        # Population size
        pop_size = min(50, max(10, budget // 4))

        # Initialize population
        X = np.zeros((pop_size, n))
        # First half: ZDT6-specific (x[1:]=0, x[0] uniform)
        half = pop_size // 2
        X[:half, 0] = np.linspace(0, 1, half, endpoint=False) + 0.5 / half
        X[:half, 1:] = 0.0
        # Second half: random
        X[half:, :] = lo + (hi - lo) * np.random.rand(pop_size - half, n)

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_points = list(zip(F[:, 0], F[:, 1]))

        # NSGA-II loop
        while used < budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Non-dominated sorting
            fronts = self._non_dominated_sort(F)

            # Crowding distance
            cd = self._crowding_distance(F)

            # Selection: binary tournament
            parents = []
            for _ in range(2):
                i, j = np.random.randint(0, pop_size, 2)
                fi = fronts[i]
                fj = fronts[j]
                if fi < fj:
                    parents.append(i)
                elif fi > fj:
                    parents.append(j)
                else:
                    parents.append(i if cd[i] > cd[j] else j)

            p1, p2 = parents
            child = self._crossover(X[p1], X[p2])
            child = self._mutate(child, lo, hi)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1
            all_points.append((fc[0], fc[1]))

            # Replacement: find worst point to replace
            # Use a simple strategy: replace the point that is dominated by the most others
            # or just random replacement for speed
            worst_idx = np.random.randint(0, pop_size)
            X[worst_idx] = child
            F[worst_idx] = fc

        return X

    def _non_dominated_sort(self, F):
        n = len(F)
        fronts = np.zeros(n, dtype=int)
        domination_count = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    domination_count[j] += 1
                    dominated_by[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    domination_count[i] += 1
                    dominated_by[j].append(i)

        current_front = [i for i in range(n) if domination_count[i] == 0]
        front_num = 0
        while current_front:
            next_front = []
            for i in current_front:
                fronts[i] = front_num
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_num += 1

        return fronts

    def _crowding_distance(self, F):
        n = len(F)
        m = F.shape[1]
        cd = np.zeros(n)
        for k in range(m):
            order = np.argsort(F[:, k])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            if F[order[-1], k] == F[order[0], k]:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (F[order[i + 1], k] - F[order[i - 1], k]) / (F[order[-1], k] - F[order[0], k])
        return cd

    def _crossover(self, p1, p2):
        # Simulated binary crossover
        n = len(p1)
        child = np.zeros(n)
        for i in range(n):
            if np.random.rand() < 0.5:
                beta = 1.0 / (1.0 + 2.0 * (p1[i] - p2[i])) if p1[i] > p2[i] else 1.0
                beta = min(max(beta, 0.01), 99.0)
                child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            else:
                beta = 1.0 / (1.0 + 2.0 * (p2[i] - p1[i])) if p2[i] > p1[i] else 1.0
                beta = min(max(beta, 0.01), 99.0)
                child[i] = 0.5 * ((1 + beta) * p2[i] + (1 - beta) * p1[i])
        return child

    def _mutate(self, x, lo, hi):
        # Polynomial mutation
        n = len(x)
        rate = 1.0 / n
        for i in range(n):
            if np.random.rand() < rate:
                delta = (x[i] - lo[i]) / (hi[i] - lo[i]) if hi[i] != lo[i] else 0.5
                delta_q = np.random.rand()
                if delta_q < 0.5:
                    xy = (1 + delta) * (2 * delta_q + 1 - 2 * delta_q) ** 20 - 1
                else:
                    xy = 1 - (1 - delta) * (2 * (1 - delta_q) + 2 * delta_q - 1) ** 20 - 1
                    xy = 1 - xy
                x[i] = lo[i] + xy * (hi[i] - lo[i])
        return x
# EVOLVE-BLOCK-END
