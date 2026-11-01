"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            # Check even sum of coordinates for D_lattice
            if sum(signs) % 2 != 0:
                continue
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
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
        """Simulated annealing with add/remove/swap/2-for-1 moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 3.0
        cooling = 0.9998
        min_T = 0.03
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        # Precompute some helper structures for speed
        # For swap moves, we need to check if removing u and adding v is valid
        # v is not in in_set, u is in in_set, and v is not a neighbor of any in_set except u
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            
            # Choose a move type
            r = rng.random()
            
            if r < 0.3:
                # ADD move: add a vertex not in set
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0 or rng.random() < np.exp(delta / T) if delta < 0 else True:
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
                    
            elif r < 0.6:
                # SWAP move: remove one in-set vertex, add one not-in-set vertex
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0 or len(not_in_set) == 0:
                    stuck_counter += 1
                    T *= cooling
                    if T < min_T:
                        T = min_T
                    continue
                    
                u = in_set_indices[rng.integers(len(in_set_indices))]
                # Find not-in-set vertices that are not neighbors of any in_set vertex except possibly u
                # A vertex v can be added if all its in_set neighbors are just u
                # i.e., count_conflicts[v] == 1 and u is a neighbor of v
                candidates = not_in_set[(count_conflicts[not_in_set] == 1) & conflicts[u][not_in_set]]
                if len(candidates) == 0:
                    stuck_counter += 1
                    T *= cooling
                    if T < min_T:
                        T = min_T
                    continue
                    
                v = candidates[rng.integers(len(candidates))]
                # Swap: remove u, add v
                # Net change: 0 (remove 1, add 1)
                # But we need to check: after removing u, can we add v?
                # v's only in_set neighbor is u, so after removing u, v has 0 in_set neighbors -> valid
                in_set[u] = False
                count_conflicts[neighbors[u]] -= 1
                in_set[v] = True
                count_conflicts[neighbors[v]] += 1
                # current_size unchanged
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                T *= cooling
                if T < min_T:
                    T = min_T
                continue
                
            elif r < 0.8:
                # 2-FOR-1 move: add 2 vertices, remove 1
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0 or len(not_in_set) < 2:
                    stuck_counter += 1
                    T *= cooling
                    if T < min_T:
                        T = min_T
                    continue
                    
                u = in_set_indices[rng.integers(len(in_set_indices))]
                # Find pairs (v1, v2) in not_in_set such that:
                # - v1 and v2 are not neighbors of each other
                # - Each of v1, v2 has at most u as its in_set neighbor
                # - v1 and v2 are not neighbors of each other
                candidates = not_in_set[(count_conflicts[not_in_set] <= 1) & conflicts[u][not_in_set]]
                if len(candidates) < 2:
                    stuck_counter += 1
                    T *= cooling
                    if T < min_T:
                        T = min_T
                    continue
                    
                # Try to find two non-adjacent candidates
                v1 = candidates[rng.integers(len(candidates))]
                v2_cands = candidates[~conflicts[v1][candidates]]
                if len(v2_cands) == 0:
                    stuck_counter += 1
                    T *= cooling
                    if T < min_T:
                        T = min_T
                    continue
                    
                v2 = v2_cands[rng.integers(len(v2_cands))]
                
                # Check: after removing u, both v1 and v2 should have 0 in_set neighbors
                # and v1, v2 not adjacent
                if conflicts[v1][v2]:
                    stuck_counter += 1
                    T *= cooling
                    if T < min_T:
                        T = min_T
                    continue
                    
                in_set[u] = False
                count_conflicts[neighbors[u]] -= 1
                in_set[v1] = True
                count_conflicts[neighbors[v1]] += 1
                in_set[v2] = True
                count_conflicts[neighbors[v2]] += 1
                current_size += 1
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                T *= cooling
                if T < min_T:
                    T = min_T
                continue
            else:
                # REMOVE move: remove a vertex from in_set (to escape local optima)
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    stuck_counter += 1
                    T *= cooling
                    if T < min_T:
                        T = min_T
                    continue
                    
                u = in_set_indices[rng.integers(len(in_set_indices))]
                in_set[u] = False
                count_conflicts[neighbors[u]] -= 1
                current_size -= 1
                stuck_counter += 1
                T *= cooling
                if T < min_T:
                    T = min_T
                continue
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            if stuck_counter > 2000:
                T = max(T, 2.5)
                stuck_counter = 0
                
        return local_best_in_set, local_best_size
    
    def local_polish(in_set_init, max_iters=8000):
        """Exact local search: try all single add moves, accept improving ones."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # Precompute count_conflicts
        count_conflicts = np.zeros(N4, dtype=np.int32)
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
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 2:
            break
        
        sa_time = min(remaining_time * 0.4, 10.0)
        if sa_time < 1.5:
            sa_time = 1.5
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Exact local polish on best found so far
        remaining_time = deadline - 4 - time.time()
        if remaining_time > 1.5:
            polish_result, polish_size = local_polish(sa_result, max_iters=5000)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
