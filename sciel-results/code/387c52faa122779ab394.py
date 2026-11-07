# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = [0] * n
        fronts = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[j] += 1
        front = [i for i in range(n) if nd[i] == 0]
        if front:
            fronts.append(front)
        k = 0
        while k < len(fronts):
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return [float('inf')] * n
        cd = [0.0] * n
        for m in range(self.n_obj):
            sorted_idx = sorted(range(n), key=lambda i: F[indices[i]][m])
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            f_min = F[indices[sorted_idx[0]]][m]
            f_max = F[indices[sorted_idx[-1]]][m]
            if f_max == f_min:
                continue
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F[indices[sorted_idx[i + 1]]][m] - F[indices[sorted_idx[i - 1]]][m]) / (f_max - f_min)
        return cd

    def _tournament(self, ranks, cd, size=2):
        i = np.random.randint(len(ranks))
        j = np.random.randint(len(ranks))
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            if cd[i] > cd[j]:
                return i
            else:
                return j

    def _crossover(self, x1, x2, lo, hi):
        """SBX-like crossover with probability 1.0, eta=20."""
        if np.random.rand() < 0.5:
            return x1.copy(), x2.copy()
        beta = 1.0 / (1.0 + 2.0 * np.random.rand())
        c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
        c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _mutation(self, x, lo, hi, rate=0.1):
        x = x.copy()
        mask = np.random.rand(self.dim) < rate
        if np.any(mask):
            x[mask] = lo[mask] + (hi[mask] - lo[mask]) * np.random.rand(np.sum(mask))
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 30s safety margin before 320s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a fraction of budget, at least 20
        N = max(20, min(50, budget // 4))
        
        # Initial population
        X = lo + (hi - lo) * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N

        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        all_indices = list(range(N))

        while used < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            
            # Assign ranks
            ranks = [0] * N
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Crowding distance per front
            cd = [0.0] * N
            for front in fronts:
                front_cd = self._crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cd[idx] = front_cd[i]

            # Generate offspring
            offspring = []
            while len(offspring) < N:
                if time.time() > deadline - 5.0:
                    break
                p1 = self._tournament(ranks, cd)
                p2 = self._tournament(ranks, cd)
                c1, c2 = self._crossover(X[p1], X[p2], lo, hi)
                c1 = self._mutation(c1, lo, hi, rate=1.0 / dim)
                c2 = self._mutation(c2, lo, hi, rate=1.0 / dim)
                offspring.append(c1)
                offspring.append(c2)
                used += 2
                if used > budget:
                    offspring = offspring[:N]
                    break

            if not offspring:
                break

            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])
            used += len(offspring)

            # Combine parent and offspring
            X_comb = np.vstack([X, np.array(offspring)])
            F_comb = np.vstack([F, F_off])
            N_comb = len(X_comb)

            # Non-dominated sorting on combined
            fronts_comb = self._fast_non_dominated_sort(F_comb)

            # Select next generation
            X_new = []
            F_new = []
            for front in fronts_comb:
                if len(X_new) + len(front) <= N:
                    X_new.extend([X_comb[i] for i in front])
                    F_new.extend([F_comb[i] for i in front])
                else:
                    # Need to sort by crowding distance
                    remaining = N - len(X_new)
                    front_cd = self._crowding_distance(F_comb, front)
                    sorted_front = sorted(front, key=lambda i: front_cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    break

            X = np.array(X_new)
            F = np.array(F_new)

            # Record all evaluated points
            for i in range(len(offspring)):
                all_X.append(offspring[i].copy())
                all_F.append(F_off[i].copy())

        # Return the best set (non-dominated points from all evaluations)
        # For hypervolume, we want the non-dominated set
        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)
        
        # Find non-dominated points
        fronts = self._fast_non_dominated_sort(all_F_arr)
        if fronts:
            best_indices = fronts[0]
            return all_X_arr[best_indices]
        return all_X_arr

# EVOLVE-BLOCK-END
