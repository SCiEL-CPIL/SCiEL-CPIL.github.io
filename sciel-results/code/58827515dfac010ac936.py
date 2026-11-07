import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = min(20, budget)
        self.archive_size = 50

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        
        # Use a fixed seed for reproducibility within the call
        rng = np.random.default_rng(42)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((self.pop_size, d))
        F = np.array([problem(x) for x in X])
        used = self.pop_size
        
        # Maintain an archive of best non-dominated points
        archive_X = X.copy()
        archive_F = F.copy()
        
        while used < self.budget:
            # Check time limit
            if time.time() > deadline:
                break
            
            # Differential Evolution approach
            # Select three distinct random parents
            i1 = rng.integers(self.pop_size)
            i2 = rng.integers(self.pop_size)
            i3 = rng.integers(self.pop_size)
            
            # Ensure they are distinct
            while i2 == i1:
                i2 = rng.integers(self.pop_size)
            while i3 == i1 or i3 == i2:
                i3 = rng.integers(self.pop_size)
            
            # DE mutation: v = x_i + F*(x_j - x_k)
            F_factor = 0.8
            v = X[i1] + F_factor * (X[i2] - X[i3])
            
            # Clip to bounds
            v = np.clip(v, lo, hi)
            
            # Crossover with current solution
            # Use the best point in the current population as the target
            # Find non-dominated points in the population
            ranks, cd = self._fast_non_dominated_sort(F)
            # Select the best non-dominated point
            front0 = np.where(ranks == 0)[0]
            if len(front0) > 0:
                # Pick randomly from the first front
                target_idx = front0[rng.integers(len(front0))]
            else:
                target_idx = rng.integers(self.pop_size)
            
            # Crossover
            CR = 0.9
            child = np.empty(d)
            j_rand = rng.integers(d)
            for j in range(d):
                if rng.random() < CR or j == j_rand:
                    child[j] = v[j]
                else:
                    child[j] = X[target_idx][j]
            
            child = np.clip(child, lo, hi)
            
            # Evaluate
            F_child = problem(child)
            used += 1
            
            # Update archive
            archive_X, archive_F = self._update_archive(archive_X, archive_F, child, F_child)
            
            # Update population using elitism
            # Combine current population with child
            X_new = np.vstack([X, child.reshape(1, -1)])
            F_new = np.vstack([F, F_child.reshape(1, -1)])
            
            # Select best pop_size points
            X, F = self._select_best(X_new, F_new, self.pop_size)
        
        # Final: combine population and archive, return best points
        all_X = np.vstack([X, archive_X])
        all_F = np.vstack([F, archive_F])
        
        # Remove dominated points
        all_X, all_F = self._remove_dominated(all_X, all_F)
        
        return all_X
    
    def _fast_non_dominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        
        # Fast non-dominated sorting
        remaining = list(range(n))
        current_rank = 0
        
        while remaining:
            front = []
            for i in remaining:
                is_dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    front.append(i)
            
            if not front:
                break
                
            for i in front:
                ranks[i] = current_rank
                remaining.remove(i)
            
            current_rank += 1
        
        return ranks, None
    
    def _update_archive(self, archive_X, archive_F, new_x, new_f):
        # Check if new point dominates any archive points
        to_remove = []
        for i in range(len(archive_F)):
            if np.all(new_f <= archive_F[i]) and np.any(new_f < archive_F[i]):
                to_remove.append(i)
        
        # Check if new point is dominated by any archive point
        is_dominated = False
        for i in range(len(archive_F)):
            if np.all(archive_F[i] <= new_f) and np.any(archive_F[i] < new_f):
                is_dominated = True
                break
        
        if is_dominated:
            return archive_X, archive_F
        
        # Remove dominated points
        if to_remove:
            mask = np.ones(len(archive_F), dtype=bool)
            mask[to_remove] = False
            archive_X = archive_X[mask]
            archive_F = archive_F[mask]
        
        # Add new point
        archive_X = np.vstack([archive_X, new_x.reshape(1, -1)])
        archive_F = np.vstack([archive_F, new_f.reshape(1, -1)])
        
        # Trim archive if too large
        if len(archive_F) > self.archive_size:
            # Keep the most diverse points
            # Use a simple grid-based thinning
            archive_X, archive_F = self._thin_archive(archive_X, archive_F, self.archive_size)
        
        return archive_X, archive_F
    
    def _thin_archive(self, archive_X, archive_F, max_size):
        if len(archive_F) <= max_size:
            return archive_X, archive_F
        
        # Simple approach: keep points with best sum of objectives
        sums = np.sum(archive_F, axis=1)
        order = np.argsort(sums)[:max_size]
        
        return archive_X[order], archive_F[order]
    
    def _select_best(self, X, F, n_select):
        if len(F) <= n_select:
            return X, F
        
        # Remove dominated points first
        X, F = self._remove_dominated(X, F)
        
        if len(F) <= n_select:
            return X, F
        
        # If still too many, use non-dominated sorting
        ranks, _ = self._fast_non_dominated_sort(F)
        
        # Select from first front
        front0 = np.where(ranks == 0)[0]
        if len(front0) >= n_select:
            # Need to select from first front only
            # Use crowding distance or just random
            selected = front0[rng.integers(len(front0), size=n_select)] if hasattr(rng, 'integers') else front0[:n_select]
            return X[selected], F[selected]
        
        # Take all of first front, then from second front
        selected = list(front0)
        for rank in range(1, max(ranks) + 1):
            front_r = np.where(ranks == rank)[0]
            remaining = n_select - len(selected)
            if remaining <= 0:
                break
            if len(front_r) <= remaining:
                selected.extend(front_r)
            else:
                selected.extend(front_r[:remaining])
                break
        
        selected = np.array(selected)
        return X[selected], F[selected]
    
    def _remove_dominated(self, X, F):
        n = len(F)
        if n == 0:
            return X, F
        
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    keep[i] = False
                    break
        
        return X[keep], F[keep]
