"""Multi-objective optimizer for RE34: Island-based MOEA with error repair."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-based multi-objective evolutionary algorithm with error repair."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        # Island setup
        n_islands = 4
        pop_size = 20
        total_init = n_islands * pop_size
        
        # Ensure we don't exceed budget
        if total_init >= self.budget:
            pop_size = max(5, self.budget // n_islands - 5)
            total_init = n_islands * pop_size

        # Initialize islands with Sobol-like spread
        # Use a simple deterministic quasi-random sequence
        def sobol_like(n, dim):
            # Simple deterministic spread using Halton-like sequence
            points = np.zeros((n, dim))
            for d in range(dim):
                base = 2 + d
                for i in range(n):
                    x = 0.0
                    f = 1.0
                    idx = i + 1
                    while idx > 0:
                        f /= base
                        x += f * (idx % base)
                        idx //= base
                    points[i, d] = x
            return points

        halton_pts = sobol_like(total_init, self.dim)
        X_init = lo + (hi - lo) * halton_pts
        F_init = np.array([problem(x) for x in X_init])
        
        n_evals = total_init
        if n_evals >= self.budget:
            return X_init

        # Split into islands
        islands_X = [X_init[i*pop_size:(i+1)*pop_size] for i in range(n_islands)]
        islands_F = [F_init[i*pop_size:(i+1)*pop_size] for i in range(n_islands)]

        # Helper: compute non-dominated front indices
        def get_nondominated_indices(F):
            n = len(F)
            if n == 0:
                return []
            is_dom = np.zeros(n, dtype=bool)
            # Vectorized dominance check
            # F[i] is dominated if exists j such that F[j] <= F[i] and F[j] != F[i]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            is_dom[i] = True
                            break
            return [i for i in range(n) if not is_dom[i]]

        # Helper: crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            
            cd = np.zeros(len(indices))
            F_sub = F[indices]
            n_obj = F_sub.shape[1]
            
            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = np.argsort(F_sub[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                f_min = F_sub[sorted_idx[0], m]
                f_max = F_sub[sorted_idx[-1], m]
                
                if f_max == f_min:
                    continue
                    
                for k in range(1, len(sorted_idx) - 1):
                    cd[sorted_idx[k]] += (F_sub[sorted_idx[k+1], m] - F_sub[sorted_idx[k-1], m]) / (f_max - f_min)
            
            return cd

        # Helper: select parent from island using tournament
        def select_parent(island_F):
            nd_idx = get_nondominated_indices(island_F)
            if not nd_idx:
                return np.random.randint(len(island_F))
            
            # Compute crowding distance for non-dominated points
            cd = crowding_distance(island_F, nd_idx)
            
            # Tournament selection: pick 2 random non-dominated, choose the one with higher CD
            i1 = nd_idx[np.random.randint(len(nd_idx))]
            i2 = nd_idx[np.random.randint(len(nd_idx))]
            
            # Map back to original index
            # cd is indexed by position in nd_idx
            pos1 = nd_idx.index(i1)
            pos2 = nd_idx.index(i2)
            
            if cd[pos1] > cd[pos2]:
                return i1
            else:
                return i2

        # Helper: DE crossover + mutation
        def de_crossover(parent1, parent2, parent3, F_parent1, lo, hi):
            # Differential Evolution crossover
            # v = parent3 + F * (parent1 - parent2)
            F_de = 0.8
            CR = 0.5
            
            v = parent3 + F_de * (parent1 - parent2)
            
            # Crossover
            mask = np.random.rand(self.dim) < CR
            # Ensure at least one gene is from v
            if not np.any(mask):
                j = np.random.randint(self.dim)
                mask[j] = True
            
            child = np.where(mask, v, parent1)
            
            # Clip to bounds
            child = np.clip(child, lo, hi)
            
            return child

        # Helper: error repair (clamp to bounds, which is already done, but let's add a small repair step)
        def error_repair(x, lo, hi):
            # If x is outside bounds, project it back
            # This is already handled by clip, but let's make it explicit
            return np.clip(x, lo, hi)

        # Main loop
        migration_interval = 10  # Migrate every 10 generations
        gen = 0
        
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            gen += 1
            
            # Select an island
            island_idx = np.random.randint(n_islands)
            
            # If island is small, fill it first
            if len(islands_X[island_idx]) < pop_size:
                # Pick a random parent from the island
                if len(islands_X[island_idx]) == 0:
                    # Pick from another island
                    other_island = np.random.choice([i for i in range(n_islands) if i != island_idx])
                    p_idx = np.random.randint(len(islands_X[other_island]))
                    parent = islands_X[other_island][p_idx].copy()
                    child = parent + 0.1 * (hi - lo) * np.random.randn(self.dim)
                    child = error_repair(child, lo, hi)
                else:
                    p_idx = np.random.randint(len(islands_X[island_idx]))
                    parent = islands_X[island_idx][p_idx].copy()
                    child = parent + 0.1 * (hi - lo) * np.random.randn(self.dim)
                    child = error_repair(child, lo, hi)
                
                f_child = problem(child)
                n_evals += 1
                
                # Add to island
                islands_X[island_idx] = np.vstack([islands_X[island_idx], child])
                islands_F[island_idx] = np.vstack([islands_F[island_idx], f_child])
                
                # Trim if over size
                if len(islands_X[island_idx]) > pop_size:
                    # Keep non-dominated, then by crowding distance
                    nd_idx = get_nondominated_indices(islands_F[island_idx])
                    if len(nd_idx) > pop_size:
                        cd = crowding_distance(islands_F[island_idx], nd_idx)
                        # Sort by CD descending
                        order = np.argsort(cd)[::-1]
                        keep = [nd_idx[i] for i in order[:pop_size]]
                        # Also add any dominated points if needed? No, just keep top CD
                        islands_X[island_idx] = islands_X[island_idx][keep]
                        islands_F[island_idx] = islands_F[island_idx][keep]
                    elif len(nd_idx) < pop_size:
                        # Keep all non-dominated, fill with best dominated? 
                        # For simplicity, just keep all if <= pop_size
                        pass
                    else:
                        islands_X[island_idx] = islands_X[island_idx][nd_idx]
                        islands_F[island_idx] = islands_F[island_idx][nd_idx]
                
                continue
            
            # Full island: perform DE evolution
            # Select 3 parents
            p1_idx = select_parent(islands_F[island_idx])
            p2_idx = select_parent(islands_F[island_idx])
            p3_idx = select_parent(islands_F[island_idx])
            
            p1 = islands_X[island_idx][p1_idx]
            p2 = islands_X[island_idx][p2_idx]
            p3 = islands_X[island_idx][p3_idx]
            
            child = de_crossover(p1, p2, p3, islands_F[island_idx][p1_idx], lo, hi)
            child = error_repair(child, lo, hi)
            
            f_child = problem(child)
            n_evals += 1
            
            # Check if child is non-dominated in the island
            # If yes, replace the worst non-dominated point (by crowding distance)
            # If no, replace a random dominated point
            
            nd_idx = get_nondominated_indices(islands_F[island_idx])
            
            # Check if child dominates any point in the island
            dominates_any = False
            dominated_indices = []
            for i in range(len(islands_F[island_idx])):
                if np.all(f_child <= islands_F[island_idx][i]) and np.any(f_child < islands_F[island_idx][i]):
                    dominates_any = True
                    dominated_indices.append(i)
            
            is_child_dominated = False
            for i in range(len(islands_F[island_idx])):
                if np.all(islands_F[island_idx][i] <= f_child) and np.any(islands_F[island_idx][i] < f_child):
                    is_child_dominated = True
                    break
            
            if not is_child_dominated:
                # Child is non-dominated
                # Add to island, then trim
                islands_X[island_idx] = np.vstack([islands_X[island_idx], child])
                islands_F[island_idx] = np.vstack([islands_F[island_idx], f_child])
                
                # Trim
                nd_idx_new = get_nondominated_indices(islands_F[island_idx])
                if len(nd_idx_new) > pop_size:
                    cd = crowding_distance(islands_F[island_idx], nd_idx_new)
                    order = np.argsort(cd)[::-1]
                    keep = [nd_idx_new[i] for i in order[:pop_size]]
                    islands_X[island_idx] = islands_X[island_idx][keep]
                    islands_F[island_idx] = islands_F[island_idx][keep]
            else:
                # Child is dominated. Replace a random dominated point in the island
                dom_indices = [i for i in range(len(islands_F[island_idx])) if i not in nd_idx]
                if dom_indices:
                    replace_idx = dom_indices[np.random.randint(len(dom_indices))]
                    islands_X[island_idx][replace_idx] = child
                    islands_F[island_idx][replace_idx] = f_child
            
            # Migration
            if gen % migration_interval == 0:
                # Each island sends its best point to a random neighbor
                for i in range(n_islands):
                    if len(islands_F[i]) == 0:
                        continue
                    nd_idx_i = get_nondominated_indices(islands_F[i])
                    if not nd_idx_i:
                        continue
                    
                    cd_i = crowding_distance(islands_F[i], nd_idx_i)
                    # Pick the one with highest CD (most diverse)
                    best_pos = np.argmax(cd_i)
                    best_idx = nd_idx_i[best_pos]
                    
                    # Send to random neighbor
                    j = np.random.choice([k for k in range(n_islands) if k != i])
                    
                    # Add to island j
                    islands_X[j] = np.vstack([islands_X[j], islands_X[i][best_idx]])
                    islands_F[j] = np.vstack([islands_F[j], islands_F[i][best_idx]])
                    
                    # Trim island j
                    nd_idx_j = get_nondominated_indices(islands_F[j])
                    if len(nd_idx_j) > pop_size:
                        cd_j = crowding_distance(islands_F[j], nd_idx_j)
                        order_j = np.argsort(cd_j)[::-1]
                        keep_j = [nd_idx_j[i] for i in order_j[:pop_size]]
                        islands_X[j] = islands_X[j][keep_j]
                        islands_F[j] = islands_F[j][keep_j]

        # Combine all islands for final result
        all_X = np.vstack([X for X in islands_X if len(X) > 0])
        return all_X
# EVOLVE-BLOCK-END
