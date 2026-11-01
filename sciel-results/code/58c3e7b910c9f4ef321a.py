import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Generate W4 vectors
    # Weight 4, entries in {-1, 0, 1}
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors = np.array(v4_candidates, dtype=np.int64)
    N = len(vectors)
    
    if N == 0:
        return np.array([], dtype=np.int64).reshape(0, D)

    # Compute conflict matrix
    # Conflict if dot > 2
    dots = vectors @ vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    # Using lists of indices for faster iteration in Python loops if needed, 
    # but NumPy boolean indexing is faster for bulk ops.
    # We will use NumPy boolean masks for the main loop.
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        # degrees: number of alive neighbors
        # Initial degrees
        degrees = np.sum(conflicts, axis=1).astype(np.int32)
        
        while np.any(alive):
            # Find alive nodes
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Get degrees of alive nodes
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            
            # Find all alive nodes with min degree
            min_cands_local = alive_indices[alive_degrees == min_deg]
            
            # Pick one randomly
            chosen = min_cands_local[rng.integers(len(min_cands_local))]
            
            in_set[chosen] = True
            
            # Remove chosen and its neighbors from alive
            to_remove = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], to_remove)))
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            # A node u loses degree for each removed neighbor v
            # degrees[u] -= number of neighbors of u in to_remove
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # conflicts[rem][:, to_remove] is a boolean matrix (len(rem) x len(to_remove))
                # Summing along axis 1 gives count of removed neighbors for each rem node
                degrees[rem] -= np.sum(conflicts[rem][:, to_remove], axis=1)
                
        return in_set

    def local_search(in_set_init, max_time=10.0):
        """Hybrid Greedy + Simulated Annealing Local Search."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 2.0
        cooling = 0.999
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N, dtype=np.int32)
        # Initialize count_conflicts
        # For each i in in_set, add 1 to count_conflicts of its neighbors
        # This is O(N^2) worst case, but N=5280, so 27M ops, fast in NumPy?
        # Better: use matrix multiplication?
        # count_conflicts = conflicts @ in_set
        count_conflicts = (conflicts @ in_set).astype(np.int32)
        
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        stuck_counter = 0
        
        while time.time() < sa_deadline:
            # Check for 0-conflict nodes (Greedy Step)
            # Nodes not in set with count_conflicts == 0
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                zero_conf_cands = not_in_set[count_conflicts[not_in_set] == 0]
                
                if len(zero_conf_cands) > 0:
                    # Add one randomly
                    v = zero_conf_cands[rng.integers(len(zero_conf_cands))]
                    
                    # Add v to set
                    in_set[v] = True
                    current_size += 1
                    
                    # Remove neighbors of v from set
                    nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                    # Note: v itself is in conflicts[v]? No, diagonal is False.
                    # But nbrs_in_set might contain v if we didn't mask it? 
                    # conflicts[v] is False at v. So v is not in nbrs_in_set.
                    
                    if len(nbrs_in_set) > 0:
                        in_set[nbrs_in_set] = False
                        current_size -= len(nbrs_in_set)
                        
                        # Update count_conflicts
                        # For each removed node u, its neighbors lose 1 conflict count
                        # For each added node v, its neighbors gain 1 conflict count
                        # 
                        # New count_conflicts = old + conflicts[v] - sum(conflicts[u] for u in removed)
                        # This is expensive to compute exactly each time.
                        # Instead, we can recompute count_conflicts periodically or use a faster update.
                        # 
                        # Fast update:
                        # count_conflicts += conflicts[v]
                        # for u in removed: count_conflicts -= conflicts[u]
                        
                        count_conflicts += conflicts[v]
                        for u in nbrs_in_set:
                            count_conflicts -= conflicts[u]
                            
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                    continue
                else:
                    stuck_counter += 1
            
            # If no 0-conflict nodes, use SA
            # Pick a random node not in set
            if len(not_in_set) == 0:
                break
                
            v = not_in_set[rng.integers(len(not_in_set))]
            
            k = count_conflicts[v]
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                # Avoid underflow
                if delta < -20 * T:
                    accept = False
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
            if accept:
                # Add v
                in_set[v] = True
                current_size += 1
                
                # Remove neighbors
                nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                if len(nbrs_in_set) > 0:
                    in_set[nbrs_in_set] = False
                    current_size -= len(nbrs_in_set)
                    
                    count_conflicts += conflicts[v]
                    for u in nbrs_in_set:
                        count_conflicts -= conflicts[u]
                        
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            if stuck_counter > 500:
                T = max(T, 1.0)
                stuck_counter = 0
            else:
                T *= cooling
                if T < min_T:
                    T = min_T
                    
        return local_best_in_set, local_best_size

    # Main Loop
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        sa_time = min(remaining_time * 0.7, 10.0)
        if sa_time < 1:
            sa_time = 1.0
            
        sa_result, sa_size = local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
            
        # 3. Polish (Second pass on best)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 5.0)
            if polish_time > 1:
                polish_result, polish_size = local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    final_indices = np.where(best_in_set)[0]
    return vectors[final_indices]

# EVOLVE-BLOCK-END
