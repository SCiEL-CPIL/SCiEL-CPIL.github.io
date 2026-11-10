import numpy as np
import itertools
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # 1. Generate all norm-3 vectors (3 non-zero entries, each +/-1)
    # Indices are combinations of 3 from 0..10
    # Signs are 8 combinations of +/-1
    vectors = []
    indices_combos = list(itertools.combinations(range(d), 3))
    signs_combos = list(itertools.product([1, -1], repeat=3))
    
    for idxs in indices_combos:
        for s in signs_combos:
            v = np.zeros(d, dtype=np.int64)
            v[idxs[0]] = s[0]
            v[idxs[1]] = s[1]
            v[idxs[2]] = s[2]
            vectors.append(v)
            
    vectors = np.array(vectors)
    n_vecs = len(vectors)
    
    # 2. Precompute dot products or just check on the fly?
    # n_vecs is 1320.
    # We can build a bitmask of compatibility.
    # compat[i] is a bitmask of all j such that dot(v_i, v_j) <= 1.
    # 1320 bits fits in 21 uint64s or a Python int.
    
    # Let's use Python integers for bitmasks.
    compat = [0] * n_vecs
    
    # Calculate compatibility
    # To speed up, we can vectorize the dot product calculation for blocks?
    # Or just loop. 1320^2 / 2 = ~870k pairs.
    # Python loop might be slow.
    # Let's use NumPy to compute the Gram matrix.
    
    G = vectors @ vectors.T
    # Condition: G[i,j] <= 1 for i != j.
    # Note: G[i,i] = 3.
    
    # Build bitmasks
    # For each i, compat[i] has bit j set if G[i,j] <= 1.
    # We can do this row by row.
    
    for i in range(n_vecs):
        mask = 0
        # Get row i
        row = G[i]
        # Find indices where row <= 1
        # Exclude i itself? The problem says "distinct points".
        # So we don't need to include i in the mask, or we can include it and handle it.
        # Let's just find j != i such that G[i,j] <= 1.
        
        # Vectorized approach for mask creation is tricky with Python ints.
        # Let's just loop j. 1320 * 1320 = 1.7M operations.
        # In Python, this might take ~1-2 seconds. Acceptable.
        
        for j in range(n_vecs):
            if i != j and row[j] <= 1:
                mask |= (1 << j)
        compat[i] = mask
        
    # 3. Greedy Clique Search
    # We want to find a large subset S such that for all i,j in S, j is in compat[i].
    # i.e., S is a clique in the compatibility graph.
    
    best_size = 0
    best_set = []
    
    # We will run a greedy heuristic multiple times.
    # Strategy:
    # 1. Shuffle vectors.
    # 2. Iterate through shuffled list.
    # 3. If vector v is compatible with all vectors in current_set, add it.
    # 4. To make it faster, we can maintain a set of "available" candidates.
    
    # A better greedy:
    # Start with all vectors available.
    # Pick a vertex v (random or highest degree).
    # Add v to clique.
    # Remove v and all incompatible vertices from available set.
    # Repeat until empty.
    # This finds a maximal clique.
    # Run multiple times with random ordering to find a large one.
    
    # To speed up "remove incompatible", we can use bitmasks.
    # available_mask: bitmask of indices still available.
    # When we pick v, new_available = available_mask & compat[v].
    # Wait, compat[v] contains all j compatible with v.
    # So new_available = available_mask & compat[v].
    
    # This is very fast with Python big integers.
    
    n_trials = 100
    trial_count = 0
    
    # Initial best is the known 220 norm-2 set?
    # We should generate that as a baseline.
    
    # Generate norm-2 set
    points_220 = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    points_220.append(v)
    points_220 = np.array(points_220)
    best_size = len(points_220)
    best_set = points_220
    
    # Now try to find better in norm-3 pool
    # We can also mix norm-2 and norm-3?
    # If we find a clique in norm-3, we can check if we can add norm-2 vectors?
    # Probably not worth the complexity. Stick to pure norm-3 first.
    
    for _ in range(n_trials):
        if time.time() > deadline:
            break
            
        # Randomize order
        indices = list(range(n_vecs))
        random.shuffle(indices)
        
        current_clique = []
        available_mask = (1 << n_vecs) - 1  # All bits set
        
        # Greedy selection
        # We can pick the first available in the shuffled list, or the one with max degree in available subgraph.
        # Picking first available is O(N^2) worst case if we scan.
        # But we can just iterate through the shuffled list and pick the first one that is in available_mask.
        # Then update available_mask.
        
        # Actually, just iterating through the shuffled list and checking if bit is set is O(N) per pick.
        # Total O(N^2). 1320^2 = 1.7M. Fast enough.
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            # Check if idx is in available_mask
            if (available_mask >> idx) & 1:
                # Add to clique
                current_clique.append(idx)
                # Update available_mask: intersect with compat[idx]
                # Note: compat[idx] does NOT include idx itself (we excluded i!=j).
                # So we don't need to clear the bit for idx, it's already cleared in the intersection?
                # Wait, if compat[idx] doesn't have bit idx, then available_mask & compat[idx] will clear bit idx.
                # Yes.
                available_mask &= compat[idx]
                
                # If available_mask becomes 0, we can break early
                if available_mask == 0:
                    break
        
        if len(current_clique) > best_size:
            best_size = len(current_clique)
            best_set = vectors[current_clique]
            
    return best_set
