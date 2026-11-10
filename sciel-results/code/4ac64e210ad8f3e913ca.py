import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    n_vectors = 2**d
    
    # Generate all 2^11 vectors in {-1, 1}^11
    # Using bit masks
    vectors = np.zeros((n_vectors, d), dtype=np.int64)
    for i in range(n_vectors):
        for j in range(d):
            if (i >> j) & 1:
                vectors[i, j] = 1
            else:
                vectors[i, j] = -1
                
    # Pre-compute Hamming distances
    # H[i, j] = number of coordinates where vectors[i] != vectors[j]
    # vectors shape (N, D)
    # diff = vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :]
    # This creates an array of size 2048x2048x11, which is ~46M elements. 
    # int64 is 8 bytes, so ~368 MB. This might be too much memory or slow to allocate.
    
    # Alternative: compute row by row or use bitwise operations.
    # Since vectors are +/- 1, we can map to 0/1 first.
    # 1 -> 1, -1 -> 0.
    # Hamming distance is popcount(xor(a, b)).
    
    bits = (vectors + 1) // 2 # 0 or 1
    
    H = np.zeros((n_vectors, n_vectors), dtype=np.int8)
    # We only need the upper triangle or full matrix.
    # Let's compute it efficiently.
    # Looping over i and vectorizing j is better for memory.
    for i in range(n_vectors):
        # xor bits[i] with all bits
        # bits[i] is shape (11,)
        # bits is shape (2048, 11)
        # xor_result = bits[i] ^ bits
        # H[i, :] = np.sum(xor_result, axis=1)
        
        # Using numpy broadcasting
        # This creates a temp array of size 2048x11
        xor_res = bits[i] ^ bits
        H[i, :] = np.sum(xor_res, axis=1)
        
    # H is symmetric, but we computed full.
    
    best_k = 0
    best_indices = None
    
    start_time = time.time()
    max_time = 580
    
    num_restarts = 20000
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        indices = np.arange(n_vectors)
        np.random.shuffle(indices)
        
        current_indices = []
        
        for idx in indices:
            if not current_indices:
                current_indices.append(idx)
                continue
                
            # Check if idx is compatible with all in current_indices
            # H[idx, current_indices] must all be >= 3
            # current_indices is a list, convert to array for indexing?
            # Or just use list comprehension? No, vectorize.
            
            # H[idx] is a vector of length N.
            # We need H[idx, j] >= 3 for all j in current_indices.
            
            # Efficient check:
            # mask = np.zeros(n_vectors, dtype=bool)
            # mask[current_indices] = True
            # if np.all(H[idx, mask] >= 3):
            
            # Even better: maintain a set of "blocked" indices?
            # Or just check against the current set.
            
            # Let's use the mask approach, it's fast enough for N=2048.
            # Creating a mask every time is O(N).
            
            # Alternative:
            # dists = H[idx, current_indices]
            # if np.min(dists) >= 3:
            
            # current_indices is a list. H[idx, current_indices] works if current_indices is a list of ints?
            # Yes, numpy accepts list of indices.
            
            dists = H[idx, current_indices]
            if np.min(dists) >= 3:
                current_indices.append(idx)
                
        if len(current_indices) > best_k:
            best_k = len(current_indices)
            best_indices = current_indices
            
    if best_indices is None:
        # Fallback
        best_indices = [0, 2**d - 1] # e.g., all -1 and all 1? 
        # 0 is all -1 (if bit 0 is LSB, 0 means all 0s in bits -> all -1s in vectors)
        # 2**d - 1 is all 1s in bits -> all 1s in vectors.
        # Hamming distance is 11. OK.
        
    best_points = vectors[best_indices]
    
    return best_points
