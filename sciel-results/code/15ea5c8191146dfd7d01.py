import numpy as np
import time

class Optimizer:
    """
    NSGA-II based multi-objective optimizer with adaptive step size and deadline guard.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop at 300s to be well within 330s limit
        deadline = start_time + 300

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        pop_size = 20
        # Initialize population using global np.random (seeded by evaluator)
        X = lo + np.random.rand(pop_size, self.dim) * range_
        F = np.array([problem(x) for x in X])
        evaluations = pop_size

        # Adaptive step size, starts at 50% of range
        sigma = 0.5

        while evaluations < self.budget:
            if time.time() > deadline:
                break

            # 1. Non-dominated sorting and selection
            # We maintain a population of size pop_size
            # Generate one child per iteration (budget constrained)
            
            # Select parent from non-dominated front
            # Find non-dominated indices in current F
            non_dom_idx = self._find_non_dominated_indices(F)
            
            if len(non_dom_idx) == 0:
                # Fallback: pick random
                parent_idx = np.random.randint(pop_size)
            else:
                # Pick random from non-dominated
                parent_idx = non_dom_idx[np.random.randint(len(non_dom_idx))]
            
            parent = X[parent_idx]
            
            # 2. Mutation
            # Gaussian mutation with adaptive sigma
            # sigma is relative to the range
            mutation = sigma * np.random.randn(self.dim)
            child = parent + mutation * range_
            child = np.clip(child, lo, hi)
            
            # 3. Evaluation
            child_f = problem(child)
            evaluations += 1
            
            # 4. Replacement
            # Combine parent population and child
            X_new = np.vstack([X, child])
            F_new = np.vstack([F, child_f])
            
            # Select next population of size pop_size using NSGA-II selection
            X, F = self._nsga2_select(X_new, F_new, pop_size)
            
            # 5. Adapt sigma
            # Decay step size
            sigma *= 0.95

        return X

    def _find_non_dominated_indices(self, F):
        """
        Returns indices of non-dominated points in F.
        """
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # Check if i dominates j
                # i dominates j if i <= j in all objectives and i < j in at least one
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    is_dominated[j] = True
        return np.where(~is_dominated)[0]

    def _nsga2_select(self, X, F, pop_size):
        """
        Selects the best pop_size individuals from X, F using NSGA-II criteria.
        """
        n = len(F)
        
        # 1. Non-dominated sorting
        ranks = np.zeros(n, dtype=int)
        remaining = np.arange(n)
        rank = 0
        
        while len(remaining) > 0:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            
            for i in front:
                ranks[i] = rank
            
            # Remove front from remaining
            front_set = set(front)
            remaining = np.array([i for i in remaining if i not in front_set])
            rank += 1
            
            # Safety break
            if rank > 50:
                break

        # 2. Crowding Distance for the last front if necessary
        # Sort indices by rank
        sorted_indices = np.argsort(ranks)
        
        selected = []
        current_rank = 0
        max_rank = ranks[sorted_indices[-1]]
        
        while len(selected) < pop_size and current_rank <= max_rank:
            # Get all points with this rank
            front_pts = [i for i in sorted_indices if ranks[i] == current_rank]
            
            if len(selected) + len(front_pts) <= pop_size:
                selected.extend(front_pts)
            else:
                # Need to select from this front based on crowding distance
                # Calculate crowding distance for these points
                F_front = F[front_pts]
                cd = self._crowding_distance(F_front)
                
                # Sort by CD descending (higher CD = more diverse)
                sorted_front = np.argsort(cd)[::-1]
                needed = pop_size - len(selected)
                selected.extend([front_pts[i] for i in sorted_front[:needed]])
                break
            
            current_rank += 1
        
        # If we still don't have enough (should not happen), fill with random
        if len(selected) < pop_size:
            remaining_indices = [i for i in range(n) if i not in selected]
            selected.extend(np.random.choice(remaining_indices, pop_size - len(selected), replace=False))
        
        return X[selected], F[selected]

    def _crowding_distance(self, F):
        """
        Calculates crowding distance for points in F.
        """
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        m = F.shape[1]
        
        for k in range(m):
            # Sort by objective k
            sorted_idx = np.argsort(F[:, k])
            
            # Assign infinite distance to extremes
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            
            # Normalize by range of this objective
            f_min = F[sorted_idx[0], k]
            f_max = F[sorted_idx[-1], k]
            f_range = f_max - f_min
            
            if f_range == 0:
                continue
                
            for i in range(1, n - 1):
                idx = sorted_idx[i]
                prev_idx = sorted_idx[i - 1]
                next_idx = sorted_idx[i + 1]
                cd[idx] += (F[next_idx, k] - F[prev_idx, k]) / f_range
        
        return cd
