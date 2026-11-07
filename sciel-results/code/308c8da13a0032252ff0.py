"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-like multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_sort(self, F):
        """Return list of fronts, each a list of indices."""
        n = len(F)
        if n == 0:
            return []
        
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
        current_front = [i for i in range(n) if n_dom[i] == 0]
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

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        m = len(F[0])
        for k in range(m):
            vals = F[indices, k]
            min_v, max_v = np.min(vals), np.max(vals)
            if max_v - min_v < 1e-15:
                continue
            sorted_idx = np.argsort(vals)
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            for i in range(1, n - 1):
                dist[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / (max_v - min_v)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population
        pop_size = min(50, self.budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track archive of non-dominated points
        archive_X = []
        archive_F = []
        
        def update_archive(x, f):
            """Add to archive if non-dominated, remove dominated points."""
            non_dominated = True
            for i in range(len(archive_F)):
                af = archive_F[i]
                if np.all(af <= f) and np.any(af < f):
                    non_dominated = False
                    break
                if np.all(f <= af) and np.any(f < af):
                    archive_X.pop(i)
                    archive_F.pop(i)
                    i -= 1
            
            if non_dominated:
                archive_X.append(x.copy())
                archive_F.append(f.copy())
        
        # Update initial archive
        for i in range(pop_size):
            update_archive(X[i], F[i])
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(F)
            
            # Select parents using tournament selection on first front
            first_front = fronts[0]
            if len(first_front) > 2:
                cd = self._crowding_distance(F, first_front)
                # Tournament of 2 based on crowding distance
                c1, c2 = rng.choice(len(first_front), 2, replace=False)
                if cd[c1] > cd[c2]:
                    p1_idx = first_front[c1]
                else:
                    p1_idx = first_front[c2]
                
                c1, c2 = rng.choice(len(first_front), 2, replace=False)
                if cd[c1] > cd[c2]:
                    p2_idx = first_front[c1]
                else:
                    p2_idx = first_front[c2]
            else:
                p1_idx = first_front[rng.integers(len(first_front))]
                p2_idx = first_front[rng.integers(len(first_front))]
            
            # SBX crossover
            child1 = X[p1_idx].copy()
            child2 = X[p2_idx].copy()
            
            for d in range(self.dim):
                if rng.random() < 0.5:
                    if abs(X[p1_idx][d] - X[p2_idx][d]) > 1e-14:
                        u = rng.random()
                        beta = 1.0 / (1.0 + 2.0 * u) if u <= 0.5 else 1.0 / (1.0 + 2.0 * (1.0 - u))
                        beta_q = beta ** 3.0
                        child1[d] = 0.5 * ((1.0 + beta_q) * X[p1_idx][d] + (1.0 - beta_q) * X[p2_idx][d])
                        child2[d] = 0.5 * ((1.0 + 1.0 / beta_q) * X[p2_idx][d] + (1.0 - 1.0 / beta_q) * X[p1_idx][d])
            
            # Polynomial mutation with adaptive rate
            progress = used / self.budget
            mut_rate = 0.1 + 0.2 * (1.0 - progress)
            eta = 20.0
            
            for d in range(self.dim):
                if rng.random() < mut_rate:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    child1[d] += delta_q * (hi[d] - lo[d])
                
                if rng.random() < mut_rate:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    child2[d] += delta_q * (hi[d] - lo[d])
            
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            
            # Evaluate children
            f1 = problem(child1)
            used += 1
            f2 = problem(child2)
            used += 1
            
            # Update archive
            update_archive(child1, f1)
            update_archive(child2, f2)
            
            # Update population using elite selection
            # Combine population and offspring
            combined_X = np.vstack([X, child1.reshape(1, -1), child2.reshape(1, -1)])
            combined_F = np.vstack([F, f1.reshape(1, -1), f2.reshape(1, -1)])
            
            # Non-dominated sort of combined population
            combined_fronts = self._non_dominated_sort(combined_F)
            
            # Select next population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_X)
                    cd = self._crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return X
# EVOLVE-BLOCK-END
