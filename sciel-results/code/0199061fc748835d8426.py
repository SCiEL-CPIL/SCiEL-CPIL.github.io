"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size: use a good fraction of budget
        pop_size = min(100, max(20, self.budget // 2))

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: fast non-dominated sorting (simplified)
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
            
            current_front = [i for i in range(n) if n_count[i] == 0]
            fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            
            return fronts

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.ones(n)
            
            dist = np.zeros(n)
            front_F = F[front_indices]
            
            for m in range(self.n_obj):
                sorted_idx = np.argsort(front_F[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = front_F[sorted_idx[-1], m] - front_F[sorted_idx[0], m]
                if f_range > 1e-15:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (front_F[sorted_idx[k+1], m] - front_F[sorted_idx[k-1], m]) / f_range
            
            return dist

        def combine_pop(X1, F1, X2, F2):
            """Combine two populations and select top pop_size."""
            X_comb = np.vstack([X1, X2])
            F_comb = np.vstack([F1, F2])
            
            fronts = non_dominated_sort(F_comb)
            
            selected_idx = []
            for front in fronts:
                if len(selected_idx) + len(front) <= pop_size:
                    selected_idx.extend(front)
                else:
                    # Need to truncate this front using crowding distance
                    need = pop_size - len(selected_idx)
                    cd = crowding_distance(F_comb, front)
                    # Sort by crowding distance descending, take top 'need'
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected_idx.extend(sorted_front[:need])
                    break
            
            return X_comb[selected_idx], F_comb[selected_idx]

        def crossover(x1, x2):
            """SBX-like crossover."""
            if rng.random() < 0.5:
                mask = rng.random(self.dim) < 0.5
                c1 = np.where(mask, x1, x2)
                c2 = np.where(mask, x2, x1)
                return c1, c2
            else:
                return x1.copy(), x2.copy()

        def mutate(x, rate=0.1):
            """Gaussian mutation."""
            x_mut = x.copy()
            mask = rng.random(self.dim) < rate
            x_mut[mask] += 0.1 * (hi - lo)[mask] * rng.standard_normal(np.sum(mask))
            return np.clip(x_mut, lo, hi)

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Tournament selection to pick parents
            def tournament_select(F, X, k=2):
                idx = rng.integers(0, len(F), size=k)
                best = idx[0]
                for i in idx[1:]:
                    if np.all(F[i] <= F[best]) and np.any(F[i] < F[best]):
                        best = i
                return X[best]

            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            batch_size = min(pop_size, self.budget - used)
            for _ in range(batch_size):
                if time.time() > deadline:
                    break
                
                p1 = tournament_select(F, X)
                p2 = tournament_select(F, X)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                
                if used < self.budget and c2 is not None:
                    offspring_X.append(c2)
                    offspring_F.append(problem(c2))
                    used += 1
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine and select
            X, F = combine_pop(X, F, offspring_X, offspring_F)

        return X
# EVOLVE-BLOCK-END
