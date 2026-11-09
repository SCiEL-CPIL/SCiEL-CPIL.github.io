"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with adaptive exploration."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use smaller population for more generations
        pop_size = 30
        
        # Latin hypercube sampling for better initial coverage
        rng = np.random.default_rng(42)
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[:, d] = (perm + 0.5) / pop_size
        X = lo + (hi - lo) * X
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best points (non-dominated)
        def is_dominated(f, F_set):
            for f_other in F_set:
                if np.all(f_other <= f) and np.any(f_other < f):
                    return True
            return False

        def get_nondominated_indices(F):
            n = len(F)
            indices = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    indices.append(i)
            return indices

        # Get initial non-dominated set
        nd_indices = get_nondominated_indices(F)
        nd_X = X[nd_indices]
        nd_F = F[nd_indices]

        # Adaptive mutation step size
        base_step = 0.15

        while used < budget:
            if time.time() > deadline:
                break

            # Select parents from non-dominated set
            n_parents = min(len(nd_X), 10)
            parent_indices = rng.choice(len(nd_X), size=n_parents, replace=False)
            
            # Generate offspring
            n_offspring = min(pop_size, budget - used)
            offspring_X = []
            offspring_F = []
            
            for _ in range(n_offspring):
                p1_idx = parent_indices[rng.integers(n_parents)]
                p2_idx = parent_indices[rng.integers(n_parents)]
                
                # Blend crossover
                alpha = rng.random()
                child = alpha * nd_X[p1_idx] + (1 - alpha) * nd_X[p2_idx]
                
                # Adaptive mutation
                step = base_step * (1.0 - 0.3 * used / budget)
                mask = rng.random(dim) < 0.5
                if np.any(mask):
                    child[mask] += step * (hi - lo) * rng.standard_normal(np.sum(mask))
                    child = np.clip(child, lo, hi)
                
                offspring_X.append(child)
                offspring_F.append(problem(child))
                used += 1
                
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine with current non-dominated set
            combined_X = np.vstack([nd_X, offspring_X])
            combined_F = np.vstack([nd_F, offspring_F])
            
            # Find new non-dominated set
            new_nd_indices = get_nondominated_indices(combined_F)
            nd_X = combined_X[new_nd_indices]
            nd_F = combined_F[new_nd_indices]

            # Limit non-dominated set size to avoid explosion
            if len(nd_X) > 50:
                # Keep points with largest crowding distance
                dist = np.zeros(len(nd_X))
                for m in range(n_obj):
                    vals = nd_F[:, m]
                    min_v, max_v = np.min(vals), np.max(vals)
                    if max_v > min_v:
                        sorted_idx = np.argsort(vals)
                        dist[sorted_idx[0]] = np.inf
                        dist[sorted_idx[-1]] = np.inf
                        for k in range(1, len(vals) - 1):
                            dist[sorted_idx[k]] += (vals[sorted_idx[k+1]] - vals[sorted_idx[k-1]]) / (max_v - min_v)
                keep_idx = np.argsort(-dist)[:50]
                nd_X = nd_X[keep_idx]
                nd_F = nd_F[keep_idx]

        return nd_X
# EVOLVE-BLOCK-END
