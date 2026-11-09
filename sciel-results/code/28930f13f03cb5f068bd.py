"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-inspired multi-objective optimizer with adaptive mutation and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Return list of fronts (indices), front 0 = non-dominated."""
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
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            sorted_idx = np.argsort(F_sub[:, m])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
            if f_range == 0:
                continue
            for k in range(1, n - 1):
                cd[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / f_range
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Initial population: larger for better front coverage
        pop_size = min(30, max(10, self.budget // 6))
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for final hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        # Adaptive mutation: start wide, shrink over time
        base_mut_scale = 0.3
        min_mut_scale = 0.05
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(F)
            
            # Selection: pick parents from best fronts
            parent_pool = []
            for front in fronts:
                parent_pool.extend(front)
                if len(parent_pool) >= pop_size // 2:
                    break
            if len(parent_pool) < 2:
                parent_pool = list(range(pop_size))
            
            # Pick two parents via tournament
            def tournament():
                candidates = rng.integers(len(parent_pool), size=2)
                c1, c2 = parent_pool[candidates[0]], parent_pool[candidates[1]]
                # Prefer lower front rank
                front_rank = {idx: i for i, fr in enumerate(fronts) for idx in fr}
                r1 = front_rank.get(c1, 999)
                r2 = front_rank.get(c2, 999)
                return c1 if r1 <= r2 else c2
            
            p1 = tournament()
            p2 = tournament()
            
            # Crossover: blend
            alpha = rng.random()
            child = alpha * X[p1] + (1 - alpha) * X[p2]
            
            # Adaptive mutation
            progress = used / self.budget
            mut_scale = base_mut_scale * (1 - progress) + min_mut_scale * progress
            child = child + mut_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point in population (by sum of objectives)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc
            
            # Occasionally add a completely random point for exploration
            if used % 20 == 0 and used < self.budget - 5:
                rand_x = lo + (hi - lo) * rng.random(self.dim)
                rand_f = problem(rand_x)
                used += 1
                # Replace second worst
                sums = np.sum(F, axis=1)
                worst1 = int(np.argmax(sums))
                sums[worst1] = np.inf
                worst2 = int(np.argmax(sums))
                X[worst2] = rand_x
                F[worst2] = rand_f
        
        return X
# EVOLVE-BLOCK-END
