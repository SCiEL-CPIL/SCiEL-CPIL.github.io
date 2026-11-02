import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Generate weight 5 vectors
    # Indices 0..10
    # Choose 5 indices
    # Assign +/- 1
    import itertools
    
    vectors = []
    for combo in itertools.combinations(range(D), 5):
        for signs in itertools.product([1, -1], repeat=5):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            vectors.append(v)
            
    vectors = np.array(vectors, dtype=np.int64)
    R_sq = 5  # norm squared
    
    best_set = []
    best_len = 0
    
    # Try a few random shuffles
    # Estimate cost: 14784 * K * 11
    # If K=1000, 1.6e8 * 11 = 1.7e9? No, vectorized numpy is fast.
    # Each check is (K, 11) subtraction and sum.
    # 14784 iterations.
    # If we do 5 shuffles, 70k checks.
    # Should fit in 140s.
    
    for trial in range(20):
        if time.time() > deadline:
            break
            
        idx = np.random.permutation(len(vectors))
        current_set = []
        
        for i in idx:
            if time.time() > deadline:
                break
                
            v = vectors[i]
            
            if not current_set:
                current_set.append(v)
                continue
                
            # Check compatibility
            # current_set is list of arrays. Convert to array for speed?
            # Better to keep as list and only convert if needed, or use a pre-allocated array?
            # Since we add one by one, converting list to array each time is O(K).
            # Total O(N*K^2).
            # 14784 * 1000^2 is too slow.
            
            # Optimization: Maintain a numpy array of the current set.
            # But resizing numpy array is slow.
            # Alternative: Use a list, and only check against the list if K is small?
            # Or, use a batch check?
            
            # Let's try maintaining a numpy array and appending.
            # np.vstack is slow.
            
            # Better approach:
            # Keep a list of vectors.
            # Periodically convert to numpy array for bulk check? No, we add one by one.
            
            # Actually, for K ~ 1000, checking 1000 vectors is fast in numpy if vectorized.
            # But doing it 14784 times...
            
            # Let's estimate:
            # 14784 * (K * 11 ops)
            # If K=1000, 14784 * 11000 = 1.6e8 ops.
            # Python loop overhead is high.
            # Numpy call overhead is ~1us.
            # 14784 * 1us = 0.015s.
            # The computation inside: (1000, 11) array ops.
            # This is very fast.
            
            current_arr = np.array(current_set)
            diffs = current_arr - v
            dist_sq = np.sum(diffs**2, axis=1)
            min_d = np.min(dist_sq)
            
            if min_d >= R_sq:
                current_set.append(v)
                
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set
            
    return np.array(best_set, dtype=np.int64)
