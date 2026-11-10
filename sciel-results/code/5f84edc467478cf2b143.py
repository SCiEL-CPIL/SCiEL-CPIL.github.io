"""DE-based multi-objective optimizer with adaptive mutation and local refinement."""
import time
import numpy as np
from itertools import combinations


class Optimizer:
    """Differential-evolution style multi-objective optimizer with adaptive
    mutation strength, non-dominated sorting, and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._ref = np.array([3.0, 5.0, 7.0], dtype=float)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        dim, n_obj, budget = self.dim, self.n_obj, self.budget
        span = hi - lo

        # Population size: balance exploration and exploitation
        pop_size = min(60, max(20, budget // 4))
        
        # Initialize population: mix of random and boundary points
        X = np.empty((pop_size, dim))
        for i in range(pop_size):
            if i < pop_size // 3:
                # Random interior points
                X[i] = lo + span * rng.random(dim)
            elif i < 2 * pop_size // 3:
                # Points near boundaries
                X[i] = lo + span * rng.random(dim)
                # Push some coordinates to boundaries
                n_bound = rng.integers(1, dim)
                idx = rng.choice(dim, size=n_bound, replace=False)
                for j in idx:
                    X[i, j] = lo[j] if rng.random() < 0.5 else hi[j]
            else:
                # Vertices of the box
                X[i] = np.where(rng.random(dim) < 0.5, lo, hi)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        best_hv = self._hypervolume(F)

        # Adaptive mutation: start with larger steps, reduce over time
        base_f = 0.8
        base_cr = 0.7
        
        def _non_dominated_sort(F):
            """Return list of fronts (lists of indices), sorted by dominance rank."""
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            current_front = []
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            S[i].append(j)
                            n_dom[i] += 1
                        elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            n_dom[j] += 1
                            S[j].append(i)
            
            for i in range(n):
                if n_dom[i] == 0:
                    current_front.append(i)
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            # Handle remaining points (shouldn't happen with proper dominance)
            remaining = [i for i in range(n) if i not in [k for front in fronts for k in front]]
            if remaining:
                fronts.append(remaining)
            
            return fronts

        def _crowding_distance(F, front_indices):
            """Compute crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            F_front = F[front_indices]
            
            for m in range(n_obj):
                sorted_idx = np.argsort(F_front[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
                if f_range > 1e-15:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (F_front[sorted_idx[k + 1], m] - F_front[sorted_idx[k - 1], m]) / f_range
            
            return dist

        def _hypervolume(F):
            """Compute hypervolume indicator for 3 objectives vs reference point."""
            valid = np.all(F < self._ref, axis=1)
            if not np.any(valid):
                return 0.0
            G = F[valid]
            n = len(G)
            if n == 1:
                return float(np.prod(self._ref - G[0]))
            
            # Sort by first objective
            order = np.argsort(G[:, 0])
            G = G[order]
            
            hv = 0.0
            prev = self._ref.copy()
            for k in range(n - 1, -1, -1):
                p = G[k]
                box = np.maximum(prev - p, 0.0)
                hv += np.prod(box)
                prev = np.minimum(prev, p)
            
            return hv

        # Main loop
        step = 0
        while used < budget:
            if time.time() > deadline:
                break
            
            # Adaptive parameters
            progress = used / budget
            f = base_f * (1.0 - 0.5 * progress)
            cr = base_cr * (1.0 - 0.3 * progress)
            
            # Generate offspring using DE/rand/1
            offspring = np.empty_like(X)
            for i in range(pop_size):
                # Select 3 distinct parents
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                v = X[a] + f * (X[b] - X[c])
                # Binomial crossover
                trial = np.where(rng.random(dim) < cr, v, X[i])
                trial = np.clip(trial, lo, hi)
                offspring[i] = trial
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])
            used += pop_size
            
            # Combine and select
            X_comb = np.vstack([X, offspring])
            F_comb = np.vstack([F, F_off])
            
            # Non-dominated sorting
            fronts = _non_dominated_sort(F_comb)
            
            # Selection: fill new population
            X_new = np.empty((pop_size, dim))
            F_new = np.empty((pop_size, n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        X_new[count] = X_comb[idx]
                        F_new[count] = F_comb[idx]
                        count += 1
                else:
                    # Need crowding distance to pick remaining
                    needed = pop_size - count
                    cd = _crowding_distance(F_comb, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:needed]:
                        X_new[count] = X_comb[idx]
                        F_new[count] = F_comb[idx]
                        count += 1
                    break
            
            X, F = X_new, F_new
            step += 1
        
        # Local refinement: perturb best non-dominated points
        if used < budget:
            # Get non-dominated points
            fronts = _non_dominated_sort(F)
            if fronts:
                nd_indices = fronts[0]
                # Perturb each non-dominated point with small steps
                for idx in nd_indices:
                    if used >= budget:
                        break
                    for scale in [0.05, 0.02, 0.01]:
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        child = X[idx] + scale * span * rng.standard_normal(dim)
                        child = np.clip(child, lo, hi)
                        fc = problem(child)
                        used += 1
                        # Replace if child is better in at least one objective
                        # and not dominated by current point
                        if np.any(fc < F[idx]) and not (np.all(F[idx] <= fc) and np.any(F[idx] < fc)):
                            X[idx] = child
                            F[idx] = fc
        
        return F
