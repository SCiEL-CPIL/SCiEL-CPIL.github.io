"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-inspired multi-objective optimizer for WFG7 with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominance_matrix(self, F):
        """Compute domination matrix: dom[j, i] is True if F[j] dominates F[i]."""
        N = F.shape[0]
        F_j = F[:, np.newaxis, :]  # (N, 1, M)
        F_i = F[np.newaxis, :, :]  # (1, N, M)
        
        le = F_j <= F_i
        lt = F_j < F_i
        
        all_le = np.all(le, axis=2)
        any_lt = np.any(lt, axis=2)
        
        dom = all_le & any_lt
        np.fill_diagonal(dom, False)
        return dom

    def _non_dominated_sort(self, F):
        """Perform fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
        N = F.shape[0]
        dom = self._dominance_matrix(F)
        
        # S[i] = set of indices dominated by i
        # n[i] = number of solutions dominating i
        S = [[] for _ in range(N)]
        n = np.zeros(N, dtype=int)
        
        for i in range(N):
            for j in range(N):
                if dom[i, j]:  # i dominates j
                    S[i].append(j)
                    n[j] += 1
        
        # First front: solutions with n[i] == 0
        front = [i for i in range(N) if n[i] == 0]
        fronts = []
        if front:
            fronts.append(front)
        
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for solutions in indices."""
        if len(indices) <= 1:
            return np.full(len(indices), np.inf)
        
        m = self.n_obj
        cd = np.zeros(len(indices))
        F_sub = F[indices]
        
        for k in range(m):
            obj = F_sub[:, k]
            min_val, max_val = obj.min(), obj.max()
            if max_val == min_val:
                cd[:] = np.inf
                continue
            
            order = np.argsort(obj)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            
            for i in range(1, len(order) - 1):
                cd[order[i]] += (obj[order[i + 1]] - obj[order[i - 1]]) / (max_val - min_val)
        
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        rng = np.random.default_rng(42)
        pop_size = 40
        
        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = self._non_dominated_sort(F)
            
            # Select parent from first front using tournament selection based on crowding distance
            if fronts:
                first_front = fronts[0]
                if len(first_front) == 1:
                    parent_idx = first_front[0]
                else:
                    cd = self._crowding_distance(F, first_front)
                    # Tournament of 2
                    i1 = rng.integers(len(first_front))
                    i2 = rng.integers(len(first_front))
                    if cd[i1] >= cd[i2]:
                        parent_idx = first_front[i1]
                    else:
                        parent_idx = first_front[i2]
            else:
                parent_idx = rng.integers(pop_size)
            
            parent = X[parent_idx]
            
            # Adaptive mutation
            progress = used / self.budget
            step_size = 0.15 * (1 - 0.5 * progress)
            
            child = parent + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Create candidate population: current + child
            X_cand = np.vstack([X, child.reshape(1, -1)])
            F_cand = np.vstack([F, fc.reshape(1, -1)])
            
            # Non-dominated sort on candidate population
            fronts_cand = self._non_dominated_sort(F_cand)
            
            # Select next population of size pop_size
            X_next = []
            F_next = []
            
            for front in fronts_cand:
                if len(X_next) + len(front) <= pop_size:
                    # Take entire front
                    for idx in front:
                        X_next.append(X_cand[idx])
                        F_next.append(F_cand[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(X_next)
                    cd = self._crowding_distance(F_cand, front)
                    # Sort by crowding distance (descending)
                    order = np.argsort(cd)[::-1]
                    for i in range(remaining):
                        idx = front[order[i]]
                        X_next.append(X_cand[idx])
                        F_next.append(F_cand[idx])
                    break
            
            X = np.array(X_next)
            F = np.array(F_next)

        return X
# EVOLVE-BLOCK-END
