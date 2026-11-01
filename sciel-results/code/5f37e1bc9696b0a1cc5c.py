"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate all candidate vectors with norm^2 in {2, 4}
    # Weight 1: one entry ±2, rest 0 -> norm^2 = 4
    # Weight 2: two entries ±1, rest 0 -> norm^2 = 2
    # Weight 4: four entries ±1, rest 0 -> norm^2 = 4
    
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_candidates.append(v)
    
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    all_candidates = v1_candidates + v2_candidates + v4_candidates
    vectors_all = np.array(all_candidates, dtype=np.int64)
    
    N = len(vectors_all)
    
    # Compute conflict matrix: conflict if dot > 2
    # (distance squared = norm_i^2 + norm_j^2 - 2*dot, need >= max_norm^2 = 4)
    # For two vectors with norms n1, n2: dist^2 = n1^2 + n2^2 - 2*dot
    # We need dist^2 >= max(norm^2) = 4
    # So: n1^2 + n2^2 - 2*dot >= 4
    # i.e., dot <= (n1^2 + n2^2 - 4) / 2
    
    norms_sq = np.sum(vectors_all ** 2, axis=1)
    dots = vectors_all @ vectors_all.T
    
    # Conflict threshold per pair: dot > (n1^2 + n2^2 - 4) / 2
    threshold = (norms_sq[:, None] + norms_sq[None, :] - 4) / 2.0
    conflicts = dots > threshold
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing local search."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 2.0
        cooling = 0.999
        min_T = 0.01
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
                
            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)
            
            # Choose from candidates with min or near-min conflict count
            threshold_count = min_count + 1
            cands = not_in_set[counts <= threshold_count]
            v = cands[rng.integers(len(cands))]
            
            k = count_conflicts[v]
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.3
            else:
                accept = rng.random() < np.exp(delta / T)
                
            if accept:
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            if stuck_counter > 800:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    def local_polish(in_set_init, max_iters=10000):
        """Exact local search: try all single moves, accept improving ones."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # Precompute count_conflicts
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 3:
            improved = False
            iters += 1
            
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            
            best_gain = 0
            best_v = -1
            
            for v in not_in_set:
                k = count_conflicts[v]
                gain = 1 - k
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v >= 0 and best_gain > 0:
                to_remove = np.where(conflicts[best_v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                in_set[best_v] = True
                current_size += best_gain
                count_conflicts[neighbors[best_v]] += 1
                improved = True
        
        return in_set, current_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 80
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Exact local polish on best found so far
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            polish_result, polish_size = local_polish(sa_result, max_iters=5000)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
    
    # Final polish on the best solution
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1:
        final_result, final_size = local_polish(best_in_set, max_iters=5000)
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_result
    
    # Final result
    final_indices = np.where(best_in_set)[0]
    result_vectors = vectors_all[final_indices]
    
    return result_vectors

# EVOLVE-BLOCK-END
