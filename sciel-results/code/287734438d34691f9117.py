import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 100.0

    # 1. Generate Base Set (252 vectors)
    base_vectors = []
    
    # Type 1 in first 8
    for i, j in itertools.permutations(range(8), 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * s1
                v[j] = 2 * s2
                base_vectors.append(v)
                
    # Type 3 in first 8 (E8 half-integer roots scaled)
    for signs in itertools.product([1, -1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            v = np.zeros(D, dtype=np.int64)
            v[:8] = signs
            base_vectors.append(v)
            
    # Type 1 in last 3
    for i, j in itertools.permutations(range(8, 11), 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * s1
                v[j] = 2 * s2
                base_vectors.append(v)
                
    S = np.array(base_vectors, dtype=np.int64)
    
    # 2. Generate Candidate Pool of Type 3 vectors
    # All permutations of 8 indices, all signs
    # To save memory/time, we process in batches of indices
    
    # Pre-compute all index combinations
    index_combos = list(itertools.combinations(range(D), 8))
    
    # For each index combo, generate all 256 sign vectors
    # Check against S
    
    for indices in index_combos:
        if time.time() > deadline:
            break
        
        # Generate 256 vectors for these indices
        # Shape (256, 8) of signs
        signs = np.array(list(itertools.product([1, -1], repeat=8)), dtype=np.int64)
        
        # Create (256, 11) array
        cands = np.zeros((256, D), dtype=np.int64)
        cands[:, indices] = signs
        
        # Check compatibility with S
        # We need min_{u in S} (u . v) <= 4 for all v in cands?
        # No, for each v in cands, we need max_{u in S} (u . v) <= 4.
        # dots = S @ cands.T  -> Shape (K, 256)
        # We need np.all(dots <= 4, axis=0)
        
        # To avoid huge memory, check in chunks if S is large
        # But S starts at 252. 252 * 256 * 8 bytes = 512KB. Fine.
        
        dots = S @ cands.T
        valid_mask = np.all(dots <= 4, axis=0)
        
        if np.any(valid_mask):
            new_vectors = cands[valid_mask]
            S = np.vstack([S, new_vectors])
            
    return S
# EVOLVE-BLOCK-END
