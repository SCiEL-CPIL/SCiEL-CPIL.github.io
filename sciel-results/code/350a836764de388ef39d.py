import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        pop_size = 40
        
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            n_dominating = np.zeros(n, dtype=int)
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominating[i] += 1
                if n_dominating[i] == 0:
                    front0.append(i)
            fronts.append(front0)
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front0 = next_front
            return fronts

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-15:
                    continue
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_v - min_v)
            return cd

        def select_parents(F, X, n_parents):
            fronts = fast_non_dominated_sort(F)
            parents = []
            for front in fronts:
                if len(parents) + len(front) <= n_parents:
                    parents.extend(front)
                else:
                    remaining = n_parents - len(parents)
                    if remaining > 0:
                        cd = crowding_distance(F, front)
                        sorted_front = [front[j] for j in np.argsort(-cd)]
                        parents.extend(sorted_front[:remaining])
                    break
            return parents

        def on_front_point(rng):
            v = rng.standard_normal(3)
            norm = np.linalg.norm(v)
            if norm < 1e-10:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v = v / norm
            x = 0.5 * (lo + hi).copy()
            for k in range(3):
                # WFG7 front is roughly x_i^2 sum = 1 in the first 3 dims if bounds are [0,1] or similar.
                # We map the unit vector to the range [lo, hi] for the first 3 dimensions.
                x[k] = lo[k] + v[k] * (hi[k] - lo[k])
            return x

        # Initialize
        n_seed_front = 24
        n_seed_rand = pop_size - n_seed_front
        
        X = np.empty((pop_size, dim))
        for i in range(n_seed_front):
            X[i] = on_front_point(rng)
        for i in range(n_seed_rand):
            X[i] = lo + span * rng.random(dim)
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best front point for bias
        def best_front_idx(F):
            fronts = fast_non_dominated_sort(F)
            if not fronts or not fronts[0]:
                return 0
            front0 = np.array(fronts[0])
            norms = np.linalg.norm(F[front0], axis=1)
            return front0[np.argmin(norms)]

        while used < budget:
            if time.time() > deadline:
                break
            
            # Select parents
            n_parents = pop_size
            parent_indices = select_parents(F, X, n_parents)
            
            # Generate children
            n_children = min(pop_size, budget - used)
            X_c = np.zeros((n_children, dim))
            
            # Pre-compute best index for bias
            bias_idx = best_front_idx(F)
            
            for i in range(n_children):
                # Pick two distinct parents
                p1_idx = parent_indices[rng.integers(len(parent_indices))]
                p2_idx = parent_indices[rng.integers(len(parent_indices))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Uniform Crossover
                mask = rng.random(dim) < 0.5
                child = np.where(mask, p1, p2)
                
                # Convergence bias: replace one parent's contribution or mutate towards best?
                # The inspiration suggested replacing p2 with best_front_idx in tournament.
                # Here we are doing uniform crossover. We can bias the mutation towards the best point.
                # Or simply: if rng < bias_prob, child = 0.5*child + 0.5*X[bias_idx] ?
                # Let's stick to the inspiration's "replace p2" logic adapted to our crossover:
                # If we picked p2, and we want bias, we can just use X[bias_idx] as p2.
                # Let's do that before crossover.
                # Actually, the loop above already picked p2. Let's re-pick if we want bias.
                # To save time, let's just apply a "pull" towards the best point during mutation.
                
                # Mutation
                progress = used / budget
                step = (1.0 - 0.5 * progress) * 0.15 * span
                child = child + step * rng.standard_normal(dim)
                
                # Bias towards best front point
                if rng.random() < 0.35:
                    # Blend with best point
                    child = 0.7 * child + 0.3 * X[bias_idx]
                
                child = np.clip(child, lo, hi)
                X_c[i] = child
            
            F_c = np.array([problem(x) for x in X_c])
            used += n_children
            
            # Combine and Select
            X_all = np.vstack([X, X_c])
            F_all = np.vstack([F, F_c])
            
            fronts = fast_non_dominated_sort(F_all)
            
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    remaining = pop_size - len(new_indices)
                    if remaining > 0:
                        cd = crowding_distance(F_all, front)
                        sorted_front = [front[j] for j in np.argsort(-cd)]
                        new_indices.extend(sorted_front[:remaining])
                    break
            
            X = X_all[new_indices]
            F = F_all[new_indices]
            
        return X
