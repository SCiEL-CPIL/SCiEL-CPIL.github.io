import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Return non-dominated set indices."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(i + 1, n):
                if dominated[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return ~dominated

    def _crowding_distance(self, F, nd_idx):
        """Compute crowding distance for non-dominated solutions."""
        if len(nd_idx) <= 2:
            return np.full(len(nd_idx), np.inf)
        dist = np.zeros(len(nd_idx))
        for k in range(self.n_obj):
            sorted_idx = nd_idx[np.argsort(F[nd_idx, k])]
            dist[np.argwhere(nd_idx == sorted_idx[0]).flatten()] = np.inf
            dist[np.argwhere(nd_idx == sorted_idx[-1]).flatten()] = np.inf
            f_min = F[nd_idx, k].min()
            f_max = F[nd_idx, k].max()
            if f_max == f_min:
                continue
            for i in range(1, len(sorted_idx) - 1):
                local_pos = np.argwhere(nd_idx == sorted_idx[i]).flatten()[0]
                dist[local_pos] += (F[sorted_idx[i + 1], k] - F[sorted_idx[i - 1], k]) / (f_max - f_min)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        pop = int(min(30, self.budget // 3))
        X = lo + span * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        
        # Main evolutionary loop
        while used < self.budget - 20:  # Reserve 20 for polish
            if time.time() >= deadline:
                break
            
            # Find non-dominated solutions
            nd_mask = self._non_dominated_sort(F)
            nd_idx = np.where(nd_mask)[0]
            
            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(np.sum(F, axis=1))])
            
            # Compute crowding distance for selection
            cd = self._crowding_distance(F, nd_idx)
            
            # Select parent: prefer high crowding distance among non-dominated
            parent_idx = nd_idx[rng.choice(len(nd_idx), p=cd / cd.sum())]
            
            # Adaptive mutation with decreasing step size
            progress = used / self.budget
            step_size = 0.15 * (1.0 - 0.6 * progress)
            
            # Mix of Gaussian and uniform mutation
            if rng.random() < 0.7:
                mutation = step_size * span * rng.standard_normal(self.dim)
            else:
                mutation = step_size * span * (rng.random(self.dim) * 2 - 1)
            
            child = X[parent_idx] + mutation
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replace worst solution based on distance to reference point
            ref = np.array([3.0, 5.0, 7.0])
            dists = np.linalg.norm(F - ref, axis=1)
            worst = int(np.argmax(dists))
            X[worst], F[worst] = child, fc
        
        # Polish phase: refine best solutions with small steps
        nd_mask = self._non_dominated_sort(F)
        nd_idx = np.where(nd_mask)[0]
        polish_candidates = nd_idx[:min(10, len(nd_idx))]
        
        for _ in range(20):
            if used >= self.budget or time.time() >= deadline:
                break
            
            # Pick a candidate to polish
            cand_idx = polish_candidates[rng.integers(len(polish_candidates))]
            base = X[cand_idx].copy()
            
            # Small local search
            best_x = base
            best_f = problem(base)
            best_f = np.array(best_f)
            
            for step in [0.02, 0.01, 0.005]:
                if used >= self.budget or time.time() >= deadline:
                    break
                for _ in range(3):
                    if used >= self.budget or time.time() >= deadline:
                        break
                    trial = base + step * span * rng.standard_normal(self.dim)
                    trial = np.clip(trial, lo, hi)
                    tf = np.array(problem(trial))
                    used += 1
                    if np.all(tf <= best_f):
                        best_x = trial
                        best_f = tf
            
            # Replace candidate with polished version
            X[cand_idx] = best_x
            F[cand_idx] = best_f
        
        return X
