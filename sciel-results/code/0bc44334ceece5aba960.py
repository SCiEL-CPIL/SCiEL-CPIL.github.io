"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # 1. Generate Type A vectors (Weight 1, value +/-2, Norm^2=4)
    # These are mutually compatible and compatible with all Type C vectors.
    type_a_vectors = []
    for i in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * sign
            type_a_vectors.append(v)
    
    type_a_array = np.array(type_a_vectors)
    n_a = len(type_a_array) # 22

    # 2. Generate Type C vectors (Weight 4, values +/-1, Norm^2=4)
    type_c_vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            type_c_vectors.append(v)
            
    type_c_array = np.array(type_c_vectors)
    n_c = len(type_c_array) # 5280

    # 3. Compute Conflict Graph for Type C vectors
    # Conflict if dot > 2
    dots = type_c_array @ type_c_array.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(n_c)]
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(n_c, dtype=bool)
    
    def run_greedy_sa(max_time):
        in_set = np.zeros(n_c, dtype=bool)
        conf_counts = np.zeros(n_c, dtype=np.int32)
        
        # Greedy Initialization: Random Order
        order = rng.permutation(n_c)
        for v in order:
            if conf_counts[v] == 0:
                in_set[v] = True
                conf_counts[neighbors[v]] += 1
                
        current_size = in_set.sum()
        
        # Local Search
        sa_deadline = time.time() + max_time
        T = 1.0
        cooling = 0.999
        
        while time.time() < sa_deadline:
            # 1. Try to add a free node
            free_nodes = np.where((~in_set) & (conf_counts == 0))[0]
            if len(free_nodes) > 0:
                v = free_nodes[rng.integers(len(free_nodes))]
                in_set[v] = True
                conf_counts[neighbors[v]] += 1
                current_size += 1
            else:
                # 2. Try a 1-1 swap
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                
                # Pick a random blocked node
                v = not_in_set[rng.integers(len(not_in_set))]
                
                # Get neighbors of v that are in the set
                nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                
                if len(nbrs_in_set) == 0:
                    if conf_counts[v] == 0:
                        in_set[v] = True
                        conf_counts[neighbors[v]] += 1
                        current_size += 1
                    continue
                
                # Pick a random neighbor to remove
                u = nbrs_in_set[rng.integers(len(nbrs_in_set))]
                
                # Simulate removal of u
                in_set[u] = False
                conf_counts[neighbors[u]] -= 1
                current_size -= 1
                
                # Now try to add v
                if conf_counts[v] == 0:
                    in_set[v] = True
                    conf_counts[neighbors[v]] += 1
                    current_size += 1
                else:
                    # Revert
                    in_set[u] = True
                    conf_counts[neighbors[u]] += 1
                    current_size += 1
                    
                # 3. Try a 2-2 swap (with lower probability)
                if rng.random() < 0.2:
                    # Pick two random nodes from not_in_set
                    if len(not_in_set) >= 2:
                        v1, v2 = rng.choice(not_in_set, size=2, replace=False)
                        # Check if v1 and v2 are compatible
                        if not conflicts[v1, v2]:
                            # Get neighbors of v1 and v2 in set
                            nbrs1 = np.where(conflicts[v1] & in_set)[0]
                            nbrs2 = np.where(conflicts[v2] & in_set)[0]
                            
                            # We need to remove at most 2 nodes to make space
                            # Simplified: If we can remove u1 from nbrs1 and u2 from nbrs2
                            # such that v1 and v2 become free.
                            
                            # This is complex to check efficiently.
                            # Skip for now to keep code simple and fast.
                            pass

                # Cooling
                T *= cooling
                if T < 0.1:
                    T = 0.1
                    
        return in_set, current_size

    # Run multiple restarts
    n_restarts = 0
    while time.time() < deadline - 5:
        n_restarts += 1
        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break
            
        # Allocate time for this restart
        # First restart gets more time, subsequent ones less?
        # Or fixed time per restart.
        t_alloc = min(remaining * 0.8, 10.0)
        
        in_set, size = run_greedy_sa(t_alloc)
        
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
    # Construct final result
    final_c_indices = np.where(best_in_set)[0]
    result_c = type_c_array[final_c_indices]
    
    # Combine Type A and Type C
    if len(result_c) > 0:
        final_vectors = np.vstack([type_a_array, result_c])
    else:
        final_vectors = type_a_array
        
    return final_vectors

# EVOLVE-BLOCK-END
