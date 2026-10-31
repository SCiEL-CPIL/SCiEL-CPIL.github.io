import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate Type A vectors: (±2, 0, ..., 0)
    type_a = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            type_a.append(v)
    type_a = np.array(type_a, dtype=np.int64)
    
    # 2. Generate Type B vectors: (±1, ±1, ±1, ±1, 0, ..., 0)
    type_b_candidates = []
    # Choose 4 positions out of 11
    for pos in itertools.combinations(range(D), 4):
        # All 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            type_b_candidates.append(v)
    type_b = np.array(type_b_candidates, dtype=np.int64)
    
    # We want to find a large subset S_B of type_b such that 
    # for all u, v in S_B, u dot v <= 2.
    # Note: u dot v > 2 implies incompatibility.
    
    best_size = 0
    best_subset_indices = []
    
    # Greedy search with random restarts
    n_restarts = 500  # Adjust based on time
    
    # Pre-compute dot product matrix for type_b? 
    # 5280 x 5280 is ~28M entries. 28M * 8 bytes = 224MB. 
    # This might be too large or slow to compute fully.
    # 5280^2 * 11 ops = 300M ops. ~1-2 seconds.
    # Let's try to compute it once.
    
    # Actually, checking compatibility on the fly during greedy might be faster if the subset is small.
    # But greedy adds vectors. 
    # Let's use a vectorized check for the current set.
    
    def greedy_select():
        # Start with empty set
        selected_indices = []
        # Shuffle candidates
        candidates = np.arange(len(type_b))
        np.random.shuffle(candidates)
        
        # Current set of vectors
        current_set = []
        
        for idx in candidates:
            if time.time() > deadline:
                break
            v = type_b[idx]
            # Check compatibility with all vectors in current_set
            # If current_set is empty, compatible.
            if not current_set:
                current_set.append(v)
                selected_indices.append(idx)
            else:
                # Compute dot products
                # current_set is list of arrays
                # Convert to array for vectorization? 
                # Or just loop. 
                # Since we expect ~500 vectors, loop is 500 * 11 = 5500 ops.
                # 5280 candidates * 5500 ops = 29M ops per restart.
                # 500 restarts * 29M = 14.5B ops. Too slow in Python.
                
                # We need a faster check.
                # Maintain a matrix or use NumPy.
                # Let's maintain a NumPy array of selected vectors.
                # But appending to NumPy array is slow.
                
                # Alternative: Keep a list, and only check against the list.
                # Optimization: If we pre-compute the incompatibility graph, we can do it faster.
                
                # Let's try a different strategy: 
                # Randomly pick a vector, add it, remove all incompatible ones from the pool.
                # This is the standard greedy MIS algorithm.
                
                pass # Placeholder for logic
        
    # Revised Strategy: 
    # 1. Pre-compute incompatibility mask? 
    #    5280x5280 bool matrix. 
    #    Dot product > 2.
    #    This takes ~2s to compute.
    #    Then greedy: 
    #    Pool = set(range(5280))
    #    Selected = []
    #    While Pool not empty and time < deadline:
    #       Pick random v from Pool
    #       Add v to Selected
    #       Remove v and all incompatible neighbors from Pool
    #    
    #    This is efficient if we can quickly find neighbors.
    #    With a boolean matrix, finding neighbors is a slice.
    
    # Let's compute the incompatibility matrix.
    # 5280 * 5280 * 11 operations.
    # Using NumPy: (type_b @ type_b.T)
    # 5280x11 @ 11x5280 = 5280x5280.
    # This is fast (~0.5s).
    
    if time.time() > deadline:
        return type_a # Fallback to seed-like result (just Type A is not the seed, seed was 220)
        # Seed was 220 vectors of type (±1, ±1, 0...). 
        # My Type A is 22 vectors. 
        # I should return the best found.
        
    try:
        # Compute dot products
        # Note: type_b has 5280 vectors.
        dot_prods = type_b @ type_b.T
        
        # Incompatible if dot > 2
        # Also, a vector is incompatible with itself (dot=4).
        # We want to find a set where all pairwise dots <= 2.
        
        # Greedy MIS on the incompatibility graph
        # Graph edges: i-j if dot[i,j] > 2.
        
        n = len(type_b)
        
        best_selected = []
        
        # Run multiple random greedy passes
        for restart in range(200):
            if time.time() > deadline:
                break
                
            pool = set(range(n))
            selected = []
            
            # To speed up, we can pick the vector with the fewest neighbors in the pool?
            # Or just random. Random is often good for this density.
            
            while pool:
                if time.time() > deadline:
                    break
                
                # Pick a random vertex from pool
                v = random.choice(list(pool))
                selected.append(v)
                
                # Find all vertices in pool incompatible with v
                # Incompatible if dot_prods[v, u] > 2
                # Note: dot_prods[v, v] = 4 > 2, so v is removed.
                
                # Get neighbors
                # mask = dot_prods[v, :] > 2
                # neighbors = np.where(mask)[0]
                # neighbors_in_pool = [u for u in neighbors if u in pool]
                
                # This list comprehension is slow.
                # Better:
                # Convert pool to array? No.
                
                # Let's use a boolean array for pool.
                # But we are in a restart loop.
                
                # Optimization: 
                # Instead of set, use a boolean mask `in_pool`.
                # And a list of active vertices.
                
                pass # Logic needs restructuring for speed
                
    except Exception:
        pass

    # Fallback: Return the seed solution (220 vectors) if we fail to find better quickly
    # But we can do better than 220 easily.
    # Let's implement a simpler, faster greedy.
    
    # Simple Greedy:
    # 1. Shuffle type_b
    # 2. Iterate. If compatible with all previously selected, add.
    # To make it fast, we check compatibility against the *last K* added vectors first? 
    # No, must be all.
    
    # Let's try the "Pick and Remove" strategy with a NumPy boolean mask for the pool.
    
    def run_greedy_pass():
        in_pool = np.ones(n, dtype=bool)
        active_count = n
        selected = []
        
        # We need to pick a random active vertex.
        # Finding a random True in a boolean array is O(N).
        # We can maintain a list of active indices.
        
        active_indices = list(range(n))
        random.shuffle(active_indices)
        
        # This is O(N^2) in the worst case if we scan.
        
        # Alternative: 
        # Just iterate through a shuffled list of all candidates.
        # If candidate is still "available" (not removed by previous picks), add it.
        # How to track "available"?
        # When we pick v, we mark all its neighbors as "unavailable".
        
        available = np.ones(n, dtype=bool)
        
        shuffled_indices = np.random.permutation(n)
        
        for idx in shuffled_indices:
            if time.time() > deadline:
                break
            if not available[idx]:
                continue
            
            # Select idx
            selected.append(idx)
            
            # Mark idx and its neighbors as unavailable
            # Neighbors are j where dot_prods[idx, j] > 2
            # Get mask
            mask = dot_prods[idx, :] > 2
            available[mask] = False
            
        return selected

    # This is O(N * N) per pass? 
    # For each selected v, we do a slice operation on dot_prods (N ops) and update available (N ops).
    # Total selected ~500.
    # 500 * 5280 = 2.6M ops per pass.
    # Very fast.
    
    best_selected = []
    
    for i in range(500):
        if time.time() > deadline:
            break
        sel = run_greedy_pass()
        if len(sel) > len(best_selected):
            best_selected = sel
            if len(best_selected) >= 496:
                break # Found optimal or near optimal

    # Construct final vector set
    # Type A + selected Type B
    selected_type_b = type_b[best_selected]
    
    final_vectors = np.vstack([type_a, selected_type_b])
    
    # Verify validity
    # Min dist squared >= 4
    # Max norm squared = 4
    # Check min dist
    # This is O(K^2 D). K ~ 518. 518^2 * 11 = 3M ops. Fast.
    if len(final_vectors) > 1:
        # Compute pairwise distances
        # Use broadcasting
        # (K, 1, D) - (1, K, D)
        # This creates a K x K x D array. 518^2 * 11 = 3M elements. 24MB. OK.
        dists_sq = np.sum((final_vectors[:, np.newaxis, :] - final_vectors[np.newaxis, :, :]) ** 2, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = np.min(dists_sq)
        
        # Max norm sq
        norms_sq = np.sum(final_vectors ** 2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        if min_dist_sq < max_norm_sq - 1e-9:
            # Invalid, fallback to seed
            # Seed: 220 vectors of type (±1, ±1, 0...)
            vectors = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        vectors.append(v)
            return np.array(vectors, dtype=np.int64)
            
    return final_vectors
# EVOLVE-BLOCK-END
