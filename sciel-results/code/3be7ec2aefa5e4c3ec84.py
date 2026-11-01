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
    N1 = len(vectors_v1)
    N_total = N4 + N1
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    # We need to check all pairs
    # To save memory/time, we can compute conflicts on the fly or precompute adjacency lists
    # N=5302, N^2 = 28M. Dot products are fast.
    
    # Let's precompute the adjacency list for speed in greedy
    # We need to know which W4 conflict with which W4, and W4 with W1.
    # We know W4 and W1 never conflict (dot <= 2).
    # So we only need W4-W4 conflicts.
    
    dots_v4 = vectors_v4 @ vectors_v4.T
    conflicts_v4 = dots_v4 > 2
    np.fill_diagonal(conflicts_v4, False)
    
    # Convert to adjacency list for fast lookup
    # neighbors[i] is a list of indices j such that v4[i] conflicts with v4[j]
    neighbors = [np.where(conflicts_v4[i])[0] for i in range(N4)]
    
    rng = np.random.default_rng(777)
    
    best_size = 0
    best_in_set_v4 = np.zeros(N4, dtype=bool)
    
    def greedy_randomized(seed):
        """Randomized greedy independent set on W4 graph."""
        local_rng = np.random.default_rng(seed)
        order = local_rng.permutation(N4)
        
        in_set = np.zeros(N4, dtype=bool)
        blocked = np.zeros(N4, dtype=bool)
        
        size = 0
        for v in order:
            if not blocked[v]:
                in_set[v] = True
                size += 1
                # Block neighbors
                if len(neighbors[v]) > 0:
                    blocked[neighbors[v]] = True
                    blocked[v] = True # Mark self as blocked to prevent re-adding (though order is unique)
                    
        return in_set, size

    def polish(in_set_v4):
        """Try to add any non-conflicting W4 vectors."""
        in_set = in_set_v4.copy()
        size = int(in_set.sum())
        blocked = np.zeros(N4, dtype=bool)
        
        # Calculate initial blocked
        for i in np.where(in_set)[0]:
            blocked[neighbors[i]] = True
            blocked[i] = True
            
        # Find unblocked
        unblocked = np.where(~blocked)[0]
        # Check if any can be added?
        # Actually, if it's unblocked, it has no conflict with current set.
        # So we can add all of them?
        # No, they might conflict with each other.
        # So we need to run greedy on the subgraph of unblocked nodes.
        
        if len(unblocked) == 0:
            return in_set, size
        
        # Greedy on unblocked
        local_rng = np.random.default_rng(12345)
        order = local_rng.permutation(len(unblocked))
        
        for idx in order:
            v = unblocked[idx]
            if not blocked[v]:
                in_set[v] = True
                size += 1
                if len(neighbors[v]) > 0:
                    blocked[neighbors[v]] = True
                    blocked[v] = True
                    
        return in_set, size

    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 5000:
        n_restarts += 1
        seed = n_restarts * 31 + 17
        in_set, size = greedy_randomized(seed)
        
        # Polish
        in_set_polished, size_polished = polish(in_set)
        
        if size_polished > best_size:
            best_size = size_polished
            best_in_set_v4 = in_set_polished.copy()
            
        # Check time
        if n_restarts % 100 == 0:
             elapsed = time.time() - start_time
             if elapsed > 50:
                 break

    # Final result
    final_v4_indices = np.where(best_in_set_v4)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
