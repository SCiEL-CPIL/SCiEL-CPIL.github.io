"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    deadline = time.time() + 110.0
    
    # We want to find a 3-(11,4,1) packing (or as close as possible).
    # Max blocks is 32 (since C(11,3) = 165, and each block uses C(4,3)=4 triples).
    # 165 // 4 = 41, but structural constraints (like Steiner systems) limit it.
    # Known max for 3-(11,4,1) is actually 32?
    # Let's check:
    # If we have 32 blocks, we use 128 triples. 37 triples left.
    # If we have 33 blocks, we use 132 triples.
    # The Johnson bound J(11,4,3) = floor(11/4 * floor(10/3 * floor(9/2))) = floor(2.75 * floor(3.33 * 4)) = floor(2.75 * 13) = floor(35.75) = 35.
    # So max could be 35?
    # Actually, a 3-(11,4,1) design does not exist for v=11.
    # The maximum size of a 3-(11,4,1) packing is 32.
    
    # Strategy:
    # 1. Generate all C(11,4) blocks.
    # 2. Use a greedy algorithm with random restarts to find the largest set of blocks with no shared triples.
    # 3. Expand each block to 16 sign vectors (all combinations of +/-1).
    # 4. Check if the resulting set of vectors is valid (min distance >= max norm).
    #    - Norm is 2.
    #    - Distance between distinct sign vectors on same support:
    #      If they differ in 1 sign, distance squared = 4. Distance = 2. OK.
    #      If they differ in 2 signs, distance squared = 8. Distance > 2. OK.
    #    - Distance between vectors on different supports:
    #      Supports intersect in at most 2 coordinates (since no shared triples).
    #      If intersection is 2:
    #        Let intersection be at i, j.
    #        v1 has signs s1_i, s1_j. v2 has signs s2_i, s2_j.
    #        Distance squared = |s1_i - s2_i|^2 + |s1_j - s2_j|^2 + sum_{k in supp1\supp2} s1_k^2 + sum_{k in supp2\supp1} s2_k^2
    #        |s1_i - s2_i|^2 is 0 if same sign, 4 if different.
    #        Min distance squared occurs when signs match on intersection.
    #        Then dist^2 = 0 + 0 + 2^2 + 2^2 = 8. Dist > 2. OK.
    #      If intersection is 1:
    #        Dist^2 = 0 (or 4) + 3 + 3 = 6 (or 10). OK.
    #      If intersection is 0:
    #        Dist^2 = 4 + 4 = 8. OK.
    #    So any 3-(11,4,1) packing yields a valid kissing configuration.
    
    # Let's implement the greedy search.
    blocks = list(itertools.combinations(range(D), 4))
    # Precompute triples for each block
    block_triples = [list(itertools.combinations(b, 3)) for b in blocks]
    
    best_count = 0
    best_blocks = []
    
    # Greedy with random restarts
    for _ in range(1000):
        if time.time() > deadline - 10:
            break
            
        used_triples = set()
        selected_blocks = []
        
        # Randomize order
        indices = list(range(len(blocks)))
        np.random.shuffle(indices)
        
        for idx in indices:
            b = blocks[idx]
            trs = block_triples[idx]
            if all(t not in used_triples for t in trs):
                selected_blocks.append(idx)
                used_triples.update(trs)
        
        if len(selected_blocks) > best_count:
            best_count = len(selected_blocks)
            best_blocks = selected_blocks
            # Heuristic: if we hit 32, we can stop early as it's likely optimal
            if best_count == 32:
                break
    
    # Generate vectors
    vectors = []
    for idx in best_blocks:
        b = blocks[idx]
        # Generate all 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, coord in enumerate(b):
                v[coord] = signs[k]
            vectors.append(v)
    
    if not vectors:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
