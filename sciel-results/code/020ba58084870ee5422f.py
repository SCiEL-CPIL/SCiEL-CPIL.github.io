"""NSGA-II style multi-objective optimizer for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _fast_non_dominated_sort(F):
        """Fast non-dominated sort. Returns list of fronts (lists of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
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
        
        current_front = [i for i in range(n) if nd[i] == 0]
        fronts.append(current_front)
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front
        
        return fronts

    @staticmethod
    def _crowding_distance(F, indices):
        """Compute crowding distance for a set of indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        m = len(indices[0]) if len(indices) > 0 else 0
        
        for k in range(m):
            vals = F[indices, k]
            sorted_idx = np.argsort(vals)
            cd_sorted = np.zeros(n)
            cd_sorted[0] = np.inf
            cd_sorted[-1] = np.inf
            for i in range(1, n - 1):
                denom = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
                if denom > 1e-15:
                    cd_sorted[i] = (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / denom
                else:
                    cd_sorted[i] = 0.0
            # Map back
            for i, si in enumerate(sorted_idx):
                cd[si] += cd_sorted[i]
        
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit with margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        pop_size = min(200, max(50, budget // 2))
        
        # Initialize population
        rng = np.random.RandomState(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point for hypervolume-like diversity
        ref = np.array([400.0, 400.0, 400.0])
        
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break
            
            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(F)
            
            # Select parents using tournament selection
            parents = []
            for _ in range(2):
                # Tournament size 2
                i, j = rng.randint(pop_size), rng.randint(pop_size)
                # Find front ranks
                rank_i = next(k for k, front in enumerate(fronts) if i in front)
                rank_j = next(k for k, front in enumerate(fronts) if j in front)
                if rank_i < rank_j:
                    parents.append(i)
                elif rank_j < rank_i:
                    parents.append(j)
                else:
                    # Same front, use crowding distance
                    front = fronts[rank_i]
                    cd = self._crowding_distance(F, front)
                    # Find positions in front
                    pos_i = front.index(i)
                    pos_j = front.index(j)
                    if cd[pos_i] > cd[pos_j]:
                        parents.append(i)
                    else:
                        parents.append(j)
            
            # Crossover (SBX-like, simplified as blend)
            beta = rng.random()
            child1 = beta * X[parents[0]] + (1 - beta) * X[parents[1]]
            child2 = (1 - beta) * X[parents[0]] + beta * X[parents[1]]
            
            # Mutation
            sigma = 0.1 * (hi - lo)
            child1 = child1 + sigma * rng.standard_normal(dim)
            child2 = child2 + sigma * rng.standard_normal(dim)
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            
            # Evaluate children
            f1 = problem(child1)
            used += 1
            f2 = problem(child2)
            used += 1
            
            if used >= budget:
                break
            
            # Combine parent and child populations
            X_comb = np.vstack([X, child1.reshape(1, -1), child2.reshape(1, -1)])
            F_comb = np.vstack([F, f1.reshape(1, -1), f2.reshape(1, -1)])
            
            # Selection
            fronts_comb = self._fast_non_dominated_sort(F_comb)
            
            new_X = []
            new_F = []
            for front in fronts_comb:
                if len(new_X) + len(front) <= pop_size:
                    new_X.extend([X_comb[i] for i in front])
                    new_F.extend([F_comb[i] for i in front])
                else:
                    # Need to select from this front based on crowding distance
                    cd = self._crowding_distance(F_comb, front)
                    remaining = pop_size - len(new_X)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for i in sorted_front[:remaining]:
                        new_X.append(X_comb[i])
                        new_F.append(F_comb[i])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return X
# EVOLVE-BLOCK-END
