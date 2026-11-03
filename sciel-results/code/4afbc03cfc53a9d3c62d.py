"""Rewritten program: bitset-based maximum clique with degree bound."""
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11

    # Generate candidates
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Dot products
    dot = candidates.astype(np.int16) @ candidates.T.astype(np.int16)
    
    # Compatibility: dot <= 2
    adj = [0] * n
    for i in range(n):
        mask = (dot[i] <= 2)
        mask[i] = False
        indices = np.where(mask)[0]
        bits = 0
        for j in indices:
            bits |= (1 << j)
        adj[i] = bits

    best_clique_size = 0
    best_clique_indices = []
    
    # Greedy initialization
    def greedy_clique():
        rem = (1 << n) - 1
        clique = []
        while rem:
            # Pick vertex with max degree in rem? 
            # For speed, pick first.
            lsb = rem & -rem
            v = lsb.bit_length() - 1
            clique.append(v)
            rem &= adj[v]
        return clique

    init = greedy_clique()
    best_clique_size = len(init)
    best_clique_indices = init

    def expand(clique, candidates_mask):
        nonlocal best_clique_size, best_clique_indices, deadline
        
        if time.time() > deadline:
            return
            
        if not candidates_mask:
            if len(clique) > best_clique_size:
                best_clique_size = len(clique)
                best_clique_indices = clique.copy()
            return
            
        # Bound 1: Count
        count = candidates_mask.bit_count()
        if len(clique) + count <= best_clique_size:
            return
            
        # Bound 2: Max Degree + 1
        # Calculate max degree in candidates_mask
        # This is O(N). 
        max_deg = 0
        m = candidates_mask
        # To speed up, we can just check a few vertices? 
        # Or just use count? 
        # Let's try to compute max degree.
        # If it's too slow, we skip it.
        
        # Optimization: If count is small, compute max degree.
        if count < 1000:
             while m:
                 lsb = m & -m
                 v = lsb.bit_length() - 1
                 deg = (adj[v] & candidates_mask).bit_count()
                 if deg > max_deg:
                     max_deg = deg
                 m ^= lsb
             if len(clique) + max_deg + 1 <= best_clique_size:
                 return
                 
        # Branch
        # Order vertices by degree descending to find large cliques faster?
        # Getting sorted list is expensive.
        # Let's just iterate.
        
        m = candidates_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # Prune: if len(clique) + 1 + (max possible in N(v)) <= best
            # Max possible in N(v) is bounded by degree of v in candidates?
            deg_v = (adj[v] & candidates_mask).bit_count()
            if len(clique) + 1 + deg_v <= best_clique_size:
                continue
                
            new_clique = clique + [v]
            new_mask = candidates_mask & adj[v]
            
            expand(new_clique, new_mask)
            
            # Update candidates_mask for next iteration?
            # We are using a local copy 'm' for iteration, so we don't need to update candidates_mask here.
            # But we need to ensure we don't pick the same v again? 
            # The while loop over 'm' handles distinct vertices.

    initial_mask = (1 << n) - 1
    expand([], initial_mask)

    if not best_clique_indices:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_clique_indices, dtype=np.int64)]
    return result.astype(np.int64)
