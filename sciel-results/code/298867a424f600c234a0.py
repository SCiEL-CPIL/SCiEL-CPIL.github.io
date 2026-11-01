"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin before 150s kill

    # 1. Generate all weight-2 integer vectors (norm squared 2)
    # These are vectors with exactly two non-zero entries, each +/-1.
    v2_candidates = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v2_candidates.append(v)
    
    vectors_v2 = np.array(v2_candidates, dtype=np.int64)
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    # All are mutually compatible (max dot 1 <= 2).
    # They are also compatible with any weight-4 vector (max dot 2 <= 2).
    
    # 2. Generate weight-4 supports
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports) # 330
    
    # 3. Find a large set of supports with pairwise intersection <= 2
    # Conflict if intersection >= 3
    # Build conflict graph
    # sup_sets[i] is the set of positions for support i
    sup_sets = [set(s) for s in supports]
    
    # Adjacency list for conflict graph
    neighbors = [set() for _ in range(n_supports)]
    for i in range(n_supports):
        for j in range(i + 1, n_supports):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # Greedy Minimum Degree Independent Set
    in_set = np.zeros(n_supports, dtype=bool)
    alive = np.ones(n_supports, dtype=bool)
    degrees = np.array([len(neighbors[i]) for i in range(n_supports)], dtype=np.int32)
    
    # To speed up, we can just iterate. 330 nodes is very small.
    # We don't even need complex heuristics.
    # Simple greedy:
    # 1. Calculate degrees.
    # 2. Pick node with min degree.
    # 3. Add to set.
    # 4. Remove node and neighbors from alive.
    # 5. Update degrees.
    
    # Since N is small, we can do this efficiently.
    alive_indices = np.arange(n_supports)
    
    while len(alive_indices) > 0:
        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[0] # Deterministic choice for reproducibility
        
        in_set[chosen] = True
        
        # Get neighbors to remove
        to_remove = set(neighbors[chosen])
        to_remove.add(chosen)
        
        # Update alive
        new_alive = []
        for idx in alive_indices:
            if idx not in to_remove:
                new_alive.append(idx)
        
        # Update degrees for remaining nodes
        # A node's degree decreases by the number of its neighbors removed
        # We can just recompute or update carefully.
        # Given small N, recomputing is fast enough.
        # But let's update.
        removed_list = list(to_remove)
        for idx in new_alive:
            # Count how many of its neighbors are in removed_list
            # neighbors[idx] is a set
            # intersection size
            deg_decrease = len(neighbors[idx].intersection(to_remove))
            degrees[idx] -= deg_decrease
            
        alive_indices = np.array(new_alive, dtype=np.int32)
        
        if time.time() > deadline - 10:
            break

    # 4. Generate W4 vectors for selected supports
    selected_supports = [supports[i] for i in range(n_supports) if in_set[i]]
    
    v4_candidates = []
    for support in selected_supports:
        # Generate all 16 sign variants
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(support, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64) if v4_candidates else np.empty((0, D), dtype=np.int64)
    
    # Combine V2 and V4
    if len(vectors_v4) > 0:
        final_vectors = np.vstack([vectors_v2, vectors_v4])
    else:
        final_vectors = vectors_v2
        
    return final_vectors

# EVOLVE-BLOCK-END
