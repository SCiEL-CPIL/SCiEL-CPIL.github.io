"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate all candidate vectors with squared norm in {2, 4}
    # Weight 1: one entry ±2, rest 0 -> norm^2 = 4
    # Weight 2: two entries ±1, rest 0 -> norm^2 = 2
    # Weight 4: four entries ±1, rest 0 -> norm^2 = 4
    
    candidates = []
    norms_sq = []
    
    # Weight 1 vectors (norm^2 = 4)
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
            norms_sq.append(4)
    
    # Weight 2 vectors (norm^2 = 2)
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(2)
    
    # Weight 4 vectors (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)
    
    vectors = np.array(candidates, dtype=np.int64)
    norms_sq_arr = np.array(norms_sq, dtype=np.int64)
    N = len(vectors)
    
    if N == 0:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    # Compute conflict matrix: conflict if ||v_i - v_j||^2 < max(||v_i||^2, ||v_j||^2)
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j
    # Conflict if ||v_i||^2 + ||v_j||^2 - 2*dot < max(||v_i||^2, ||v_j||^2)
    # i.e., 2*dot > min(||v_i||^2, ||v_j||^2)
    
    dots = vectors @ vectors.T
    min_norms_sq = np.minimum(norms_sq_arr[:, None], norms_sq_arr[None, :])
    conflicts = (2 * dots) > min_norms_sq
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
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
        """Min-conflict simulated annealing with 2-for-1 swap moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
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
            
            # Decide move type: single add (70%) or 2-for-1 swap (30%)
            use_swap = rng.random() < 0.3
            
            if use_swap and len(not_in_set) >= 2:
                # 2-for-1 swap: add two non-conflicting vertices, remove one
                v1 = not_in_set[rng.integers(len(not_in_set))]
                k1 = count_conflicts[v1]
                
                # Find a second vertex that doesn't conflict with v1
                candidates_v2 = not_in_set[~conflicts[v1][not_in_set]]
                if len(candidates_v2) == 0:
                    use_swap = False
                else:
                    v2 = candidates_v2[rng.integers(len(candidates_v2))]
                    k2 = count_conflicts[v2]
                    
                    # Combined conflicts
                    combined_conflicts = conflicts[v1] | conflicts[v2]
                    to_remove = np.where(combined_conflicts & in_set)[0]
                    removal_count = len(to_remove)
                    
                    delta = 2 - removal_count
                    
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        accept = rng.random() < 0.5
                    else:
                        accept = rng.random() < np.exp(delta / T)
                    
                    if accept:
                        if len(to_remove) > 0:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                        
                        in_set[v1] = True
                        in_set[v2] = True
                        count_conflicts[neighbors[v1]] += 1
                        count_conflicts[neighbors[v2]] += 1
                        
                        # Fix double-counting for shared neighbors
                        shared = np.where(conflicts[v1] & conflicts[v2])[0]
                        for u in shared:
                            count_conflicts[u] -= 1
                        
                        current_size += delta
                        
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                    else:
                        stuck_counter += 1
            else:
                # Single add move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
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
            
            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    def local_polish(in_set_init, max_iters=5000):
        """Exact local search: try all single moves and 2-for-1 swaps, accept improving ones."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            
            best_gain = 0
            best_move = None  # (type, v1, v2)
            
            # Try single adds
            for v in not_in_set:
                k = count_conflicts[v]
                gain = 1 - k
                if gain > best_gain:
                    best_gain = gain
                    best_move = ('single', v, -1)
            
            # Try 2-for-1 swaps
            if len(not_in_set) >= 2:
                for i in range(len(not_in_set)):
                    v1 = not_in_set[i]
                    k1 = count_conflicts[v1]
                    
                    for j in range(i + 1, len(not_in_set)):
                        v2 = not_in_set[j]
                        if conflicts[v1, v2]:
                            continue
                        
                        k2 = count_conflicts[v2]
                        combined = conflicts[v1] | conflicts[v2]
                        to_remove = np.where(combined & in_set)[0]
                        removal_count = len(to_remove)
                        
                        gain = 2 - removal_count
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('swap', v1, v2)
            
            if best_move is not None and best_gain > 0:
                if best_move[0] == 'single':
                    v = best_move[1]
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    
                    in_set[v] = True
                    current_size += best_gain
                    count_conflicts[neighbors[v]] += 1
                else:
                    v1, v2 = best_move[1], best_move[2]
                    combined = conflicts[v1] | conflicts[v2]
                    to_remove = np.where(combined & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    
                    in_set[v1] = True
                    in_set[v2] = True
                    current_size += best_gain
                    count_conflicts[neighbors[v1]] += 1
                    count_conflicts[neighbors[v2]] += 1
                    
                    shared = np.where(conflicts[v1] & conflicts[v2])[0]
                    for u in shared:
                        count_conflicts[u] -= 1
                
                improved = True
        
        return in_set, current_size
    
    # Multiple restarts with varied seeds
    n_restarts = 0
    max_restarts = 80
    base_seed = 42
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        rng = np.random.default_rng(base_seed + n_restarts * 137)
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Exact local polish
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            polish_result, polish_size = local_polish(sa_result, max_iters=3000)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
    
    # Final result
    final_indices = np.where(best_in_set)[0]
    result_vectors = vectors[final_indices]
    
    return result_vectors

# EVOLVE-BLOCK-END
