import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG7 with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s/330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n = self.dim
        budget = self.budget

        # Population size: larger to catch disconnected fronts
        pop_size = min(50, max(20, budget // 4))
        
        # Initial population: spread across bounds
        X = lo + span * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """True if a dominates b (a better in all, strictly better in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Returns list of fronts, each a list of indices."""
            m = len(F)
            S = [[] for _ in range(m)]  # S[i] = indices dominated by i
            n_dom = np.zeros(m, dtype=int)  # number of solutions dominating i
            fronts = [[]]
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)
            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            cd = np.zeros(m)
            for d in range(self.n_obj):
                vals = F[indices, d]
                idx_sorted = np.argsort(vals)
                cd[idx_sorted[0]] = np.inf
                cd[idx_sorted[-1]] = np.inf
                val_range = vals[idx_sorted[-1]] - vals[idx_sorted[0]]
                if val_range > 1e-15:
                    for k in range(1, m - 1):
                        cd[idx_sorted[k]] += (vals[idx_sorted[k+1]] - vals[idx_sorted[k-1]]) / val_range
            return cd

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sort into fronts
            fronts = fast_non_dominated_sort(F)
            
            # Select parents from first few fronts
            parent_pool = []
            for front in fronts[:3]:
                parent_pool.extend(front)
                if len(parent_pool) >= pop_size // 2:
                    break
            if len(parent_pool) < 2:
                parent_pool = list(range(pop_size))
            
            # Binary tournament selection (two parents)
            def tournament():
                a = parent_pool[rng.integers(len(parent_pool))]
                b = parent_pool[rng.integers(len(parent_pool))]
                # Compare by front rank then crowding
                fa = next(i for i, fr in enumerate(fronts) if a in fr)
                fb = next(i for i, fr in enumerate(fronts) if b in fr)
                if fa != fb:
                    return a if fa < fb else b
                return a if rng.random() < 0.5 else b
            
            p1 = tournament()
            p2 = tournament()
            
            # Adaptive mutation: start wide, shrink
            progress = used / budget
            base_step = 0.15 * (1.0 - 0.7 * progress)  # 0.15 -> 0.045
            step = base_step * rng.exponential(1.0)  # heavy-tailed for exploration
            
            # Crossover: blend
            alpha = rng.random()
            child = alpha * X[p1] + (1 - alpha) * X[p2]
            # Mutation
            child = child + step * span * rng.standard_normal(n)
            # Occasional larger jump to escape local regions
            if rng.random() < 0.05:
                child = child + 0.3 * span * rng.standard_normal(n)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replacement: add child, remove worst if over pop_size
            X = np.vstack([X, child.reshape(1, -1)])
            F = np.vstack([F, fc.reshape(1, -1)])
            
            if len(X) > pop_size:
                # Find which to remove: pick from last front, lowest crowding
                fronts = fast_non_dominated_sort(F)
                # Remove from the last front
                last_front = fronts[-1]
                cd = crowding_distance(F, last_front)
                worst_in_front = last_front[np.argmin(cd)]
                # Also consider removing a dominated point with low contribution
                X = np.delete(X, worst_in_front, axis=0)
                F = np.delete(F, worst_in_front, axis=0)

        return F
