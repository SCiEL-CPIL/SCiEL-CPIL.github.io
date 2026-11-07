"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary optimizer with NSGA-II-like selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        dominated_count = np.zeros(n, dtype=int)
        dominates_list = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominates_list[i].append(j)
                    dominated_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominates_list[j].append(i)
                    dominated_count[i] += 1
        
        current_front = [i for i in range(n) if dominated_count[i] == 0]
        front_num = 0
        for i in current_front:
            ranks[i] = front_num
        front_num += 1
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominates_list[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        ranks[j] = front_num
                        next_front.append(j)
            current_front = next_front
            front_num += 1
        
        return ranks

    def _crowding_distance(self, F, indices):
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        F_sub = F[indices]
        for m in range(self.n_obj):
            min_val = F_sub[:, m].min()
            max_val = F_sub[:, m].max()
            if max_val - min_val < 1e-12:
                dist[:] = np.inf
                continue
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for i in range(1, n - 1):
                dist[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / (max_val - min_val)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        # Population size
        pop_size = min(50, max(20, self.budget // 4))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            ranks = self._non_dominated_sort(F)
            
            # Crowding distance per front
            crowding = np.zeros(len(F))
            for r in np.unique(ranks):
                front_idx = np.where(ranks == r)[0]
                crowding[front_idx] = self._crowding_distance(F, front_idx)
            
            # Tournament selection: pick 2 parents from non-dominated front
            # Prefer first front, then by crowding distance
            front0_idx = np.where(ranks == 0)[0]
            if len(front0_idx) == 0:
                front0_idx = np.arange(len(F))
            
            # Select parents from first front using crowding
            p1_idx = front0_idx[rng.integers(len(front0_idx))]
            p2_idx = front0_idx[rng.integers(len(front0_idx))]
            
            # Crossover (SBX-like)
            alpha = 20
            child1 = np.zeros(self.dim)
            child2 = np.zeros(self.dim)
            for d in range(self.dim):
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (alpha + 1))
                else:
                    beta = (2 * (1 - u)) ** (1 / (alpha + 1))
                child1[d] = 0.5 * ((1 + beta) * X[p1_idx, d] + (1 - beta) * X[p2_idx, d])
                child2[d] = 0.5 * ((1 - beta) * X[p1_idx, d] + (1 + beta) * X[p2_idx, d])
            
            # Mutation
            mut_rate = 1.0 / self.dim
            for d in range(self.dim):
                if rng.random() < mut_rate:
                    sigma = 0.2 * (hi[d] - lo[d])
                    child1[d] += sigma * rng.standard_normal()
                if rng.random() < mut_rate:
                    sigma = 0.2 * (hi[d] - lo[d])
                    child2[d] += sigma * rng.standard_normal()
            
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            
            # Evaluate children
            fc1 = problem(child1)
            used += 1
            if used >= self.budget or time.time() > deadline:
                # Only add if we have budget
                if used <= self.budget:
                    # Replace worst individual
                    worst_idx = int(np.argmax(ranks + 1e6 * (1.0 / (1.0 + crowding))))
                    X[worst_idx] = child1
                    F[worst_idx] = fc1
                break
            
            fc2 = problem(child2)
            used += 1
            
            # Replace two worst individuals
            # Score: rank + penalty for low crowding
            scores = ranks.astype(float) + 10.0 / (1.0 + crowding)
            worst_indices = np.argsort(scores)[-2:]
            
            X[worst_indices[0]] = child1
            F[worst_indices[0]] = fc1
            X[worst_indices[1]] = child2
            F[worst_indices[1]] = fc2
        
        return X
# EVOLVE-BLOCK-END
