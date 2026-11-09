import time
import numpy as np

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate scaled E8 roots (240 vectors)
    # Embedded in first 8 dimensions of R^11
    
    e8_vectors = []
    
    # Type 1: 112 vectors with two +/-2 and six 0s
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    e8_vectors.append(v)
                    
    # Type 2: 128 vectors with all +/-1, even number of -1s
    # There are 2^8 = 256 total combinations.
    # We need those with even number of -1s.
    # This is equivalent to product == 1 (since -1 * -1 = 1, odd number of -1s gives -1)
    # Or simply count the -1s.
    for bits in range(256):
        v = np.zeros(8, dtype=np.int64)
        minus_count = 0
        for i in range(8):
            if (bits >> i) & 1:
                v[i] = -1
                minus_count += 1
            else:
                v[i] = 1
        if minus_count % 2 == 0:
            e8_vectors.append(v)
            
    # 2. Generate scaled A3 roots (12 vectors)
    # Embedded in last 3 dimensions of R^11 (indices 8, 9, 10)
    
    a3_vectors = []
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    a3_vectors.append(v)
                    
    # 3. Combine into R^11 vectors
    
    selected = []
    
    # Add E8 vectors (padded with 0s in last 3 dims)
    for v in e8_vectors:
        full_v = np.zeros(D, dtype=np.int64)
        full_v[:8] = v
        selected.append(full_v)
        
    # Add A3 vectors (padded with 0s in first 8 dims)
    for v in a3_vectors:
        full_v = np.zeros(D, dtype=np.int64)
        full_v[8:] = v
        selected.append(full_v)
        
    selected = np.array(selected, dtype=np.int64)
    
    # 4. Validation
    # Check min pairwise distance >= max norm
    
    if len(selected) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    # Calculate norms
    norms_sq = np.sum(selected**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Calculate min pairwise distance squared
    # k=252, O(k^2) is ~63k operations, very fast
    min_dist_sq = np.inf
    for i in range(len(selected)):
        diffs = selected[i+1:] - selected[i]
        dists_sq = np.sum(diffs**2, axis=1)
        if len(dists_sq) > 0:
            min_d = np.min(dists_sq)
            if min_d < min_dist_sq:
                min_dist_sq = min_d
                
    # Condition: min_dist_sq >= max_norm_sq
    if min_dist_sq < max_norm_sq:
        # Fallback: This should not happen, but if it does, return the E8 part only (240 vectors)
        # which is valid on its own
        return np.array(e8_vectors, dtype=np.int64).reshape(-1, 8) if False else np.array([], dtype=np.int64).reshape(0, D)
        # Actually, better fallback: just the seed's 220?
        # But we know 252 is valid.
        pass
        
    return selected

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
