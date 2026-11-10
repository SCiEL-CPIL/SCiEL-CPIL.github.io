"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-model differential evolution with crowding distance for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Storage for all evaluations
        X = []
        F = []

        def evaluate(x):
            if len(X) >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            X.append(x.copy())
            F.append(f.copy())
            return f

        # Non-dominated set computation
        def get_non_dominated_indices(F_arr):
            m = len(F_arr)
            if m == 0:
                return np.array([], dtype=int)
            if m == 1:
                return np.array([0])
            # Vectorized dominance check
            # F_arr[i] is dominated by F_arr[j] if F_arr[j] <= F_arr[i] in all and < in at least one
            # Use broadcasting
            # For each pair (i, j), check if j dominates i
            # This is O(m^2) which is fine for m up to ~200
            nd_mask = np.ones(m, dtype=bool)
            for i in range(m):
                if not nd_mask[i]:
                    continue
                for j in range(m):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0]

        # Crowding distance computation
        def crowding_distance(F_arr, indices):
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            cd = np.zeros(m)
            for k in range(n_obj):
                obj_vals = F_arr[indices, k]
                min_val = obj_vals.min()
                max_val = obj_vals.max()
                if max_val - min_val < 1e-15:
                    continue
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, m - 1):
                    cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)
            return cd

        # Phase 1: Initial diverse sampling using Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(count, dim):
            points = np.zeros((count, dim))
            for d in range(dim):
                base = d + 2
                for i in range(count):
                    points[i, d] = halton(i + 1, base)
            return points

        # Initial population: 40 Halton points + 8 corners
        n_init = 40
        init_pts = halton_sequence(n_init, n) * span + lo
        for x in init_pts:
            evaluate(x)

        # Add corners
        for i in range(2 ** n):
            pt = np.array([lo[j] if (i >> j) & 1 else hi[j] for j in range(n)])
            evaluate(pt)

        # Phase 2: Island-model DE with crowding distance
        n_islands = 3
        pop_per_island = 15
        total_pop = n_islands * pop_per_island

        # Create islands from current non-dominated points
        F_arr = np.array(F)
        nd_idx = get_non_dominated_indices(F_arr)
        
        # If not enough non-dominated points, fill with best points
        if len(nd_idx) < total_pop:
            # Sort all points by sum of objectives (simple scalarization)
            scores = F_arr.sum(axis=1)
            sorted_all = np.argsort(scores)
            extra = [i for i in sorted_all if i not in nd_idx]
            nd_idx = np.concatenate([nd_idx, extra[:total_pop - len(nd_idx)]])
        
        # Shuffle and distribute to islands
        rng.shuffle(nd_idx)
        islands = []
        for i in range(n_islands):
            start = i * pop_per_island
            end = min(start + pop_per_island, len(nd_idx))
            islands.append(nd_idx[start:end].tolist())

        # DE parameters
        F_factor = 0.8
        CR = 0.9

        # Main loop: evolve islands
        while len(X) < budget:
            if time.time() > deadline:
                break

            F_arr = np.array(F)
            
            # For each island, generate offspring using DE
            for island_id in range(n_islands):
                if len(X) >= budget:
                    break
                if time.time() > deadline:
                    break
                
                island = islands[island_id]
                if len(island) < 3:
                    # Not enough members for DE, use random points
                    n_rand = min(pop_per_island, budget - len(X))
                    rand_pts = lo + span * rng.random((n_rand, n))
                    for rp in rand_pts:
                        evaluate(rp)
                    continue

                # Compute crowding distance within island
                island_F = F_arr[island]
                cd = crowding_distance(island_F, np.arange(len(island)))
                
                # Sort by crowding distance (descending) to prefer diverse points
                sorted_local = np.argsort(-cd)
                parents = [island[idx] for idx in sorted_local[:min(len(island), 10)]]

                # Generate offspring
                n_offspring = min(pop_per_island, budget - len(X))
                for _ in range(n_offspring):
                    if len(X) >= budget or time.time() > deadline:
                        break
                    
                    # Pick three distinct parents
                    if len(parents) >= 3:
                        idxs = rng.choice(len(parents), size=3, replace=False)
                        a, b, c = parents[idxs[0]], parents[idxs[1]], parents[idxs[2]]
                    else:
                        a = b = c = parents[0]
                    
                    x_a = F_arr[a]  # Wait, we need X, not F
                    # Correct: use X array
                    x_a = np.array(X[a])
                    x_b = np.array(X[b])
                    x_c = np.array(X[c])
                    
                    # DE mutation
                    x_new = x_a + F_factor * (x_b - x_c)
                    
                    # Crossover
                    mask = rng.random(n) < CR
                    x_new = np.where(mask, x_new, x_a)
                    
                    # Small random perturbation
                    x_new += rng.standard_normal(n) * 0.03 * span
                    x_new = np.clip(x_new, lo, hi)
                    
                    evaluate(x_new)

            # Periodic migration: exchange best members between islands
            if len(X) % 30 < n_islands:  # Every ~30 evaluations
                F_arr = np.array(F)
                nd_idx = get_non_dominated_indices(F_arr)
                if len(nd_idx) >= n_islands * 2:
                    # Rebuild islands with fresh non-dominated points
                    cd_all = crowding_distance(F_arr[nd_idx], np.arange(len(nd_idx)))
                    sorted_nd = nd_idx[np.argsort(-cd_all)]
                    
                    # Redistribute
                    rng.shuffle(sorted_nd)
                    for i in range(n_islands):
                        start = i * pop_per_island
                        end = min(start + pop_per_island, len(sorted_nd))
                        if end > start:
                            islands[i] = sorted_nd[start:end].tolist()

            # Occasionally add random points for exploration
            if len(X) < budget - 10 and time.time() < deadline:
                if rng.random() < 0.1:  # 10% chance per iteration
                    n_rand = min(5, budget - len(X))
                    rand_pts = lo + span * rng.random((n_rand, n))
                    for rp in rand_pts:
                        if len(X) >= budget or time.time() > deadline:
                            break
                        evaluate(rp)

        return X
# EVOLVE-BLOCK-END
