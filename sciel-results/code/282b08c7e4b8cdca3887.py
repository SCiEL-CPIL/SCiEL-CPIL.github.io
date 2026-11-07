import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # 10s safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Smaller population for faster iterations, more generations
        pop_size = 10
        
        # Initialize with random points
        X = np.random.uniform(lo, hi, size=(pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best hypervolume point for restarts
        ref_point = np.array([3.0, 5.0, 7.0])
        
        def fast_non_dominated_sort(F):
            """Fast non-dominated sort returning fronts as list of index arrays."""
            n = len(F)
            if n == 0:
                return []
            
            # Compute domination relations
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            fronts = []
            current_front = np.where(n_dom == 0)[0].tolist()
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            return fronts

        def hypervolume_contribution(F, ref):
            """Compute hypervolume for 3 objectives. F is (n,3), ref is (3,)."""
            n = len(F)
            if n == 0:
                return 0.0
            
            # Filter points better than reference
            mask = np.all(F < ref, axis=1)
            F = F[mask]
            n = len(F)
            if n == 0:
                return 0.0
            
            # Sort by first objective
            idx = np.argsort(F[:, 0])
            F = F[idx]
            
            hv = 0.0
            for i in range(n):
                # For each point, compute the volume it uniquely contributes
                # The contribution is the box from F[i] to ref, minus boxes already counted
                # Simple approach: for 3D, use inclusion-exclusion or grid-based
                # Faster: sort by obj0, then for each point compute area in obj1-obj2 plane
                # where this point is the minimum in obj0
                
                # Volume contribution: (ref[0]-F[i,0]) * area_in_2d
                # where area_in_2d is the area of the region in obj1-obj2 
                # where no other point with smaller obj0 exists
                
                # For simplicity and speed, use a simple approximation:
                # Each point contributes (ref[0]-F[i,0]) * (ref[1]-F[i,1]) * (ref[2]-F[i,2])
                # This overcounts but is fast. For selection, relative values matter.
                hv += (ref[0] - F[i, 0]) * (ref[1] - F[i, 1]) * (ref[2] - F[i, 2])
            
            return hv

        def select_by_hv(F, ref, k):
            """Select k points with best hypervolume contribution. Returns indices."""
            n = len(F)
            if n <= k:
                return np.arange(n)
            
            # Compute individual contributions (approximate)
            mask = np.all(F < ref, axis=1)
            scores = np.zeros(n)
            scores[mask] = (ref[0] - F[mask, 0]) * (ref[1] - F[mask, 1]) * (ref[2] - F[mask, 2])
            
            # Sort by score descending
            idx = np.argsort(scores)[::-1]
            return idx[:k]

        # Main Loop: DE with fast selection
        F_ref = np.array([3.0, 5.0, 7.0])
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring using DE
            # Use DE/rand/1
            r1 = np.random.randint(0, pop_size)
            r2 = np.random.randint(0, pop_size)
            r3 = np.random.randint(0, pop_size)
            
            # Ensure distinct indices if possible
            attempts = 0
            while r1 == r2 or r2 == r3 or r1 == r3:
                r1 = np.random.randint(0, pop_size)
                r2 = np.random.randint(0, pop_size)
                r3 = np.random.randint(0, pop_size)
                attempts += 1
                if attempts > 10:
                    break
            
            v = X[r1] + 0.7 * (X[r2] - X[r3])
            v = np.clip(v, lo, hi)
            
            fv = problem(v)
            used += 1
            
            # Combine and select
            X_new = np.vstack([X, v])
            F_new = np.vstack([F, fv])
            
            # Fast selection: keep best pop_size by hypervolume contribution
            # First, filter out dominated points quickly
            # Use non-dominated sort, then select by HV
            fronts = fast_non_dominated_sort(F_new)
            
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    remaining = pop_size - len(selected)
                    # Select from this front by HV score
                    F_front = F_new[front]
                    mask = np.all(F_front < F_ref, axis=1)
                    if np.sum(mask) >= remaining:
                        # Select best from non-dominated points better than ref
                        scores = (F_ref[0] - F_front[mask, 0]) * (F_ref[1] - F_front[mask, 1]) * (F_ref[2] - F_front[mask, 2])
                        best_local = np.argsort(scores)[::-1][:remaining]
                        selected.extend(front[i] for i in best_local)
                    else:
                        # Take all better than ref, then fill with others
                        better = [front[i] for i in range(len(front)) if np.all(F_new[front[i]] < F_ref)]
                        selected.extend(better)
                        if len(selected) < pop_size:
                            others = [i for i in front if i not in selected]
                            selected.extend(others[:pop_size - len(selected)])
                    break
            
            if len(selected) < pop_size:
                # Fill with remaining
                remaining_needed = pop_size - len(selected)
                all_indices = list(range(len(F_new)))
                for i in all_indices:
                    if i not in selected and len(selected) < pop_size:
                        selected.append(i)
            
            X = X_new[selected]
            F = F_new[selected]
        
        return X
