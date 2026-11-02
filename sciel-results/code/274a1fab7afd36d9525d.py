import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    candidates = []
    
    # Type 1: One ±2, rest 0 → 2*11 = 22 vectors
    # These are the vectors 2*e_i and -2*e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Two ±1's, rest 0 → C(11,2) * 2^2 = 55 * 4 = 220 vectors
    # These are the vectors s1*e_a + s2*e_b
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # The maximum independent set for this specific graph structure is known:
    # 1. All Type 2 vectors (indices 22 to 241) are mutually compatible.
    # 2. All Type 1 vectors (indices 0 to 21) are compatible with all Type 2 vectors.
    # 3. Type 1 vectors conflict only with their sign-flipped counterpart.
    #    Specifically, vector 2*e_i (index 2*i) conflicts with -2*e_i (index 2*i+1).
    #    There are 11 such pairs.
    #    We can pick at most one from each pair.
    
    # So we pick all 220 Type 2 vectors.
    selected_indices = list(range(22, 242))
    
    # And we pick 11 Type 1 vectors, one from each pair.
    # We pick the positive ones (indices 0, 2, 4, ..., 20).
    for i in range(0, 22, 2):
        selected_indices.append(i)
        
    result = candidates[selected_indices].astype(np.int64)
    
    # Verify validity
    n = len(result)
    if n == 0:
        return candidates[:22].astype(np.int64)
    
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    # Vectorized check
    diffs = result[:, np.newaxis, :] - result[np.newaxis, :, :]
    diff_norms_sq = np.sum(diffs**2, axis=2)
    if np.min(diff_norms_sq[np.triu_indices(n, k=1)]) < max_norm_sq:
        # Invalid, fallback to just Type 2
        result = candidates[22:242].astype(np.int64)
        
    return result
