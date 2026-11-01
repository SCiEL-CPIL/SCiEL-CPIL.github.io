"""Combined program for kissing number in dimension 11.
Uses the efficient greedy MIS from inspirations plus a local-improvement (swap)
refinement, and a deterministic angular-sector seeding idea to get a good
initial independent set quickly.
"""
import numpy as np
import itertools
import time

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D."""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def generate_weight_1_vectors():
    """Generate all weight-1 vectors in {-2, 0, 2}^D (norm^2 = 4)."""
    candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def greedy_mis(conflicts, deadline, rng):
    """
    Greedy Maximum Independent Set with degree-based ordering.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    in_set = np.zeros(N, dtype=bool)
    
    while np.any(alive):
        if time.time() > deadline:
            break
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        # Pick the alive vertex with minimum degree
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        in_set[chosen] = True
        nbrs = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], nbrs)))
        alive[to_remove] = False
        
        # Update degrees for remaining alive vertices
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
    
    return in_set

def sa_local_search(in_set_init, conflicts, deadline, rng):
    """Min-conflict simulated annealing local search."""
    N = len(conflicts)
    in_set = in_set_init.copy()
    current_size = in_set.sum()
    
    T = 3.0
    cooling = 0.9995
    min_T = 0.05
    
    # count_conflicts[v] = number of neighbors of v that are in in_set
    count_conflicts = np.zeros(N, dtype=np.int32)
    for i in np.where(in_set)[0]:
        count_conflicts[np.where(conflicts[i])[0]] += 1
    
    stuck_counter = 0
    local_best_size = current_size
    local_best_in_set = in_set.copy()
    
    while time.time() < deadline:
        not_in_set = np.where(~in_set)[0]
        if len(not_in_set) == 0:
            break
        
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
                    count_conflicts[np.where(conflicts[u])[0]] -= 1
            
            in_set[v] = True
            current_size += delta
            count_conflicts[np.where(conflicts[v])[0]] += 1
            
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    vectors_v4 = generate_weight_4_vectors()
    
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    vectors_v1 = generate_weight_1_vectors()
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    # For weight-4 vectors, norm^2 = 4, so max_norm_sq = 4
    # dist_sq = 4 + 4 - 2*dot = 8 - 2*dot
    # Need dist_sq >= 4 => 8 - 2*dot >= 4 => 2*dot <= 4 => dot <= 2
    # So conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_mis(conflicts, deadline - 5, rng)
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.6, 15.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, conflicts, time.time() + sa_time, rng)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Second SA pass on best found so far (polish)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 10.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, conflicts, time.time() + polish_time, rng)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
