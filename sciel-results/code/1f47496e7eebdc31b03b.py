"""Multi-objective optimizer for DTLZ4 using structured search and NSGA-II style selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using grid-based initialization and NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate a Halton sequence value."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def _quasi_random_points(self, n, dim, lo, hi):
        """Generate quasi-random points using Halton sequence."""
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                points[i, d] = self._halton(i + 1, bases[d])
        return lo + (hi - lo) * points

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting on objective matrix F."""
        n = len(F)
        if n == 0:
            return []
        
        # S[i] = set of solutions dominated by i
        S = [[] for _ in range(n)]
        # n_dom[i] = number of solutions dominating i
        n_dom = np.zeros(n, dtype=int)
        # Fronts
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        # First front
        current_front = [i for i in range(n) if n_dom[i] == 0]
        if not current_front:
            return [list(range(n))]
        
        fronts.append(current_front)
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for solutions in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        F_front = F[front_indices]
        
        for m in range(F_front.shape[1]):
            sorted_idx = np.argsort(F_front[:, m])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
            if f_range > 0:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (F_front[sorted_idx[i + 1], m] - F_front[sorted_idx[i - 1], m]) / f_range
        
        return cd

    def _select_parents(self, F, fronts, n_parents):
        """Select parents using tournament selection based on fronts and crowding."""
        n = len(F)
        if n == 0:
            return []
        
        # Assign rank to each solution
        rank = np.zeros(n, dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
        
        # Compute crowding distance for each front
        cd = np.zeros(n)
        for front in fronts:
            front_cd = self._crowding_distance(F, front)
            for i, idx in enumerate(front):
                cd[idx] = front_cd[i]
        
        # Tournament selection
        parents = []
        for _ in range(n_parents):
            # Pick 2 random candidates
            a = np.random.randint(n)
            b = np.random.randint(n)
            # Compare by rank first, then crowding distance
            if rank[a] < rank[b]:
                parents.append(a)
            elif rank[b] < rank[a]:
                parents.append(b)
            else:
                # Same rank, pick by crowding distance
                if cd[a] > cd[b]:
                    parents.append(a)
                else:
                    parents.append(b)
        
        return parents

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population using quasi-random sampling
        init_size = min(50, self.budget // 4)
        X = self._quasi_random_points(init_size, self.dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = init_size
        
        # Keep track of all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        # NSGA-II style evolution
        pop_size = 30
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Compute fronts and crowding
            fronts = self._non_dominated_sort(F)
            
            # Select parents
            n_parents = min(20, self.budget - used)
            parent_indices = self._select_parents(F, fronts, n_parents)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            for pi in parent_indices:
                # Check time
                if time.time() > deadline:
                    break
                if used + len(offspring_F) >= self.budget:
                    break
                
                parent = X[pi]
                # Adaptive mutation
                mutation_rate = 1.0 / self.dim
                sigma = 0.1
                child = parent + sigma * (hi - lo) * np.random.standard_normal(self.dim)
                # Random bit flip for some dimensions
                flip_mask = np.random.random(self.dim) < mutation_rate
                child[flip_mask] = lo[flip_mask] + (hi[flip_mask] - lo[flip_mask]) * np.random.random(np.sum(flip_mask))
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Record offspring
            all_X.append(offspring_X)
            all_F.append(offspring_F)
            
            # Combine parent and offspring populations
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Selection: pick best pop_size from combined
            combined_fronts = self._non_dominated_sort(combined_F)
            
            # Build new population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select some from this front using crowding
                    remaining = pop_size - len(new_X)
                    front_cd = self._crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(range(len(front)), key=lambda i: front_cd[i], reverse=True)
                    for i in sorted_front[:remaining]:
                        idx = front[i]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            if len(new_X) == 0:
                break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return X
# EVOLVE-BLOCK-END
