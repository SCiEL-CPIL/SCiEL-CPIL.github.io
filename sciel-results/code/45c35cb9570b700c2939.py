"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE21: NSGA-II style with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 300.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_ok():
            return time.time() < deadline

        pop_size = 10
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    fronts[0].append(i)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            F_front = F[front_indices]
            for m in range(self.n_obj):
                order = np.argsort(F_front[:, m])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                f_range = F_front[order[-1], m] - F_front[order[0], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        dist[order[i]] += (F_front[order[i + 1], m] - F_front[order[i - 1], m]) / f_range
            return dist

        def evaluate_point(x):
            nonlocal used
            f = problem(x)
            used += 1
            return f

        while used < self.budget and time_ok():
            fronts = non_dominated_sort(F)
            # Tournament selection from front 0
            front0 = fronts[0]
            parent_idx = front0[rng.integers(len(front0))]
            
            # Mutation: Gaussian perturbation
            sigma = 0.1
            child = X[parent_idx] + sigma * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            if not time_ok():
                break
                
            fc = evaluate_point(child)
            
            # Replace worst point in last front or random point
            last_front = fronts[-1]
            if len(last_front) > 1:
                # Replace random point in last front
                replace_idx = last_front[rng.integers(len(last_front))]
            else:
                replace_idx = rng.integers(pop_size)
                
            X[replace_idx] = child
            F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END
