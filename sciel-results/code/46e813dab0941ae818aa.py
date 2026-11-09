# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 100

    def _dominance_check(self, f1, f2):
        """Check if f1 dominates f2."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting on F. Returns list of fronts (lists of indices)."""
        n = len(F)
        fronts = []
        remaining = list(range(n))
        
        while remaining:
            current_front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if self._dominance_check(F[j], F[i]):
                        dominated = True
                        break
                if not dominated:
                    current_front.append(i)
            
            if not current_front:
                current_front = remaining
            
            fronts.append(current_front)
            remaining = [i for i in remaining if i not in current_front]
            
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Calculate crowding distance for points in a front."""
        n = len(front_indices)
        if n <= 2:
            return [float('inf')] * n
        
        distances = np.zeros(n)
        for k in range(self.n_obj):
            # Sort indices by objective k
            sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i]][k])
            
            # Assign infinite distance to extremes
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            
            # Calculate range
            f_min = F[front_indices[sorted_idx[0]]][k]
            f_max = F[front_indices[sorted_idx[-1]]][k]
            
            if f_max == f_min:
                continue
                
            for i in range(1, n - 1):
                prev_idx = sorted_idx[i - 1]
                next_idx = sorted_idx[i + 1]
                distances[sorted_idx[i]] += (F[front_indices[next_idx]][k] - F[front_indices[prev_idx]][k]) / (f_max - f_min)
        
        return distances

    def _crossover(self, x1, x2, bounds, rng):
        """Simple Blend Crossover (BLX-alpha)."""
        if rng.random() > 0.9:
            return x1.copy(), x2.copy()
        
        alpha = 0.5
        x_c1 = x1.copy()
        x_c2 = x2.copy()
        
        for i in range(self.dim):
            lo = min(x1[i], x2[i])
            hi = max(x1[i], x2[i])
            spread = hi - lo
            x_c1[i] = lo - alpha * spread * rng.random()
            x_c2[i] = hi + alpha * spread * rng.random()
            
        x_c1 = np.clip(x_c1, [b[0] for b in bounds], [b[1] for b in bounds])
        x_c2 = np.clip(x_c2, [b[0] for b in bounds], [b[1] for b in bounds])
        
        return x_c1, x_c2

    def _mutation(self, x, bounds, rng, eta=20.0):
        """Polynomial Mutation."""
        x_mut = x.copy()
        for i in range(self.dim):
            if rng.random() < 0.9:
                lo = bounds[i][0]
                hi = bounds[i][1]
                if hi == lo:
                    continue
                
                u = rng.random()
                if u < 0.5:
                    delta1 = (2 * u) ** (1.0 / (eta + 1)) - 1
                    x_mut[i] += delta1 * (hi - lo)
                else:
                    delta2 = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    x_mut[i] += delta2 * (hi - lo)
                    
                x_mut[i] = np.clip(x_mut[i], lo, hi)
        return x_mut

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        
        all_X = []
        all_F = []
        used = 0
        
        # Initial Population
        X = np.array([lo + (hi - lo) * rng.random(self.dim) for _ in range(self.pop_size)])
        F = np.array([np.array(problem(x), dtype=float) for x in X])
        for x in X: all_X.append(x.copy())
        for f in F: all_F.append(f.copy())
        used += self.pop_size
        
        # Main Loop
        while used < self.budget and time.time() < deadline:
            # 1. Non-dominated sort and crowding distance for current population
            fronts = self._non_dominated_sort(F)
            
            # Assign ranks
            ranks = np.zeros(self.pop_size, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Calculate crowding distance for each front
            crowding = np.zeros(self.pop_size)
            for front in fronts:
                if len(front) > 0:
                    cd = self._crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowding[idx] = cd[i]
            
            # 2. Generate Offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < self.pop_size and used < self.budget:
                # Binary Tournament
                i1 = rng.integers(self.pop_size)
                i2 = rng.integers(self.pop_size)
                
                if ranks[i1] < ranks[i2] or (ranks[i1] == ranks[i2] and crowding[i1] > crowding[i2]):
                    p1 = X[i1]
                else:
                    p1 = X[i2]
                    
                i3 = rng.integers(self.pop_size)
                i4 = rng.integers(self.pop_size)
                
                if ranks[i3] < ranks[i4] or (ranks[i3] == ranks[i4] and crowding[i3] > crowding[i4]):
                    p2 = X[i3]
                else:
                    p2 = X[i4]
                
                # Crossover
                c1, c2 = self._crossover(p1, p2, bounds, rng)
                
                # Mutation
                c1 = self._mutation(c1, bounds, rng)
                c2 = self._mutation(c2, bounds, rng)
                
                offspring_X.append(c1)
                offspring_F.append(np.array(problem(c1), dtype=float))
                used += 1
                all_X.append(c1.copy())
                all_F.append(offspring_F[-1].copy())
                
                if used >= self.budget or time.time() > deadline:
                    break
                    
                offspring_X.append(c2)
                offspring_F.append(np.array(problem(c2), dtype=float))
                used += 1
                all_X.append(c2.copy())
                all_F.append(offspring_F[-1].copy())
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # 3. Combine and Select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort combined
            combined_fronts = self._non_dominated_sort(combined_F)
            
            new_X = []
            new_F = []
            
            for front in combined_fronts:
                if len(new_X) + len(front) <= self.pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    front_indices = front
                    cd = self._crowding_distance(combined_F, front_indices)
                    sorted_front = sorted(range(len(front_indices)), key=lambda i: cd[i], reverse=True)
                    
                    remaining_slots = self.pop_size - len(new_X)
                    for i in range(remaining_slots):
                        orig_idx = front_indices[sorted_front[i]]
                        new_X.append(combined_X[orig_idx])
                        new_F.append(combined_F[orig_idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
        return all_X
# EVOLVE-BLOCK-END
