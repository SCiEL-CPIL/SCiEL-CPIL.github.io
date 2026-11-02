import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate weight-1 vectors (entry ±2 in 1 position)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    # Strategy: Find a large set of 4-position supports with pairwise intersection <= 2.
    # For each support, include all 16 sign patterns.
    # This ensures no conflicts between vectors of different supports (since they share <= 2 positions).
    # Vectors of the same support share 4 positions, max dot product 2, so they are compatible.
    
    supports = list(itertools.combinations(range(D), 4))
    N_sup = len(supports) # 330
    
    # Build conflict graph for supports: conflict if intersection size >= 3
    sup_conflicts = np.zeros((N_sup, N_sup), dtype=bool)
    for i in range(N_sup):
        set_i = set(supports[i])
        for j in range(i + 1, N_sup):
            set_j = set(supports[j])
            if len(set_i.intersection(set_j)) >= 3:
                sup_conflicts[i, j] = True
                sup_conflicts[j, i] = True
    
    # Find Max Independent Set in support graph using Greedy + Local Search
    rng = np.random.default_rng(42)
    
    # Precompute neighbor lists for supports
    sup_neighbors = [np.where(sup_conflicts[i])[0] for i in range(N_sup)]
    sup_neighbor_counts = np.array([len(n) for n in sup_neighbors], dtype=np.int32)
    
    best_sup_size = 0
    best_sup_set = np.zeros(N_sup, dtype=bool)
    
    def greedy_sup():
        in_set = np.zeros(N_sup, dtype=bool)
        alive = np.ones(N_sup, dtype=bool)
        degrees = sup_neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = sup_neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= sup_conflicts[rem][:, to_remove].sum(axis=1)
        
        return in_set
    
    def local_search_sup(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        T = 1.0
        stuck_counter = 0
        sa_deadline = time.time() + max_time
        
        count_conflicts = np.zeros(N_sup, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[sup_neighbors[i]] += 1
        
        not_in_set = list(np.where(~in_set)[0])
        pos_in_not_in_set = {v: idx for idx, v in enumerate(not_in_set)}
        
        def remove_from_not_in_set(v):
            if v in pos_in_not_in_set:
                idx = pos_in_not_in_set[v]
                last = not_in_set[-1]
                not_in_set[idx] = last
                pos_in_not_in_set[last] = idx
                not_in_set.pop()
                del pos_in_not_in_set[v]
        
        def add_to_not_in_set(v):
            if v not in pos_in_not_in_set:
                pos_in_not_in_set[v] = len(not_in_set)
                not_in_set.append(v)
        
        while time.time() < sa_deadline:
            if len(not_in_set) == 0:
                break
            
            v = int(not_in_set[rng.integers(len(not_in_set))])
            k = count_conflicts[v]
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < np.exp(delta / T)
            
            if accept:
                to_remove = np.where(sup_conflicts[v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[sup_neighbors[u]] -= 1
                        add_to_not_in_set(int(u))
                
                in_set[v] = True
                current_size += delta
                count_conflicts[sup_neighbors[v]] += 1
                remove_from_not_in_set(v)
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            if stuck_counter > 100:
                T = max(T, 1.5)
                stuck_counter = 0
            T *= 0.999
            if T < 0.1:
                T = 0.1
        
        return local_best_in_set, local_best_size

    # Run search on supports
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 20:
        n_restarts += 1
        in_set = greedy_sup()
        current_size = int(in_set.sum())
        
        if current_size > best_sup_size:
            best_sup_size = current_size
            best_sup_set = in_set.copy()
        
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 1:
            break
        
        sa_time = min(remaining_time * 0.5, 5.0)
        sa_result, sa_size = local_search_sup(in_set, max_time=sa_time)
        
        if sa_size > best_sup_size:
            best_sup_size = sa_size
            best_sup_set = sa_result.copy()

    # Construct final vectors
    final_vectors_list = []
    
    # Add weight-1 vectors
    for v in vectors_v1:
        final_vectors_list.append(v)
        
    # Add weight-4 vectors for selected supports
    for i in np.where(best_sup_set)[0]:
        support = supports[i]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(support, signs):
                v[pos] = s
            final_vectors_list.append(v)
            
    final_vectors = np.array(final_vectors_list, dtype=np.int64)
    
    return final_vectors

# EVOLVE-BLOCK-END
