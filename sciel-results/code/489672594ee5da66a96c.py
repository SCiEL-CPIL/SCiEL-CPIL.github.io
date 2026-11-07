"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with time guard for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        m = self.n_obj
        for k in range(m):
            obj_vals = F[indices, k]
            sorted_idx = np.argsort(obj_vals)
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            f_min = obj_vals[sorted_idx[0]]
            f_max = obj_vals[sorted_idx[-1]]
            if f_max > f_min:
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (f_max - f_min)
        return dist

    def _binary_tournament(self, ranks, dist, n_select, rng):
        """Binary tournament selection based on rank and crowding distance."""
        selected = []
        n = len(ranks)
        for _ in range(n_select):
            i = rng.integers(n)
            j = rng.integers(n)
            if ranks[i] < ranks[j]:
                selected.append(i)
            elif ranks[j] < ranks[i]:
                selected.append(j)
            else:
                if dist[i] > dist[j]:
                    selected.append(i)
                else:
                    selected.append(j)
        return selected

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Initial population
        pop_size = min(40, self.budget // 5)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_F = [F.copy()]
        
        while used < self.budget:
            if time.time() > deadline - 5:
                break
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # Non-dominated sort for selection
            fronts = self._non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Crowding distance for each front
            dist = np.zeros(len(F))
            for front in fronts:
                front_dist = self._crowding_distance(F, front)
                for i, idx in enumerate(front):
                    dist[idx] = front_dist[i]
            
            # Generate offspring via binary tournament + mutation
            while len(offspring_X) < pop_size and used + len(offspring_X) < self.budget:
                parents = self._binary_tournament(ranks, dist, 2, rng)
                p1, p2 = parents[0], parents[1]
                
                # Simulated binary crossover (SBX)
                if rng.random() < 0.9:
                    eta = 15.0
                    child1 = np.zeros(self.dim)
                    child2 = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta + 1))
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                            child1[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                            child2[d] = 0.5 * ((1 - beta) * X[p1, d] + (1 + beta) * X[p2, d])
                        else:
                            child1[d] = X[p1, d]
                            child2[d] = X[p2, d]
                else:
                    child1 = X[p1].copy()
                    child2 = X[p2].copy()
                
                # Polynomial mutation
                eta_m = 20.0
                for d in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        u = rng.random()
                        if u <= 0.5:
                            delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                        child1[d] += delta * (hi[d] - lo[d])
                    if rng.random() < 1.0 / self.dim:
                        u = rng.random()
                        if u <= 0.5:
                            delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                        child2[d] += delta * (hi[d] - lo[d])
                
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                
                offspring_X.append(child1)
                offspring_X.append(child2)
                f1 = problem(child1)
                f2 = problem(child2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                used += 2
                all_F.append(np.array([f1]))
                all_F.append(np.array([f2]))
                
                if time.time() > deadline - 5:
                    break
            
            if len(offspring_X) == 0:
                break
            
            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array(offspring_F[:pop_size])
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort on combined
            fronts = self._non_dominated_sort(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    front_dist = self._crowding_distance(combined_F, front)
                    sorted_by_dist = np.argsort(front_dist)[::-1][:remaining]
                    for i in sorted_by_dist:
                        idx = front[i]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
        
        return X
# EVOLVE-BLOCK-END
