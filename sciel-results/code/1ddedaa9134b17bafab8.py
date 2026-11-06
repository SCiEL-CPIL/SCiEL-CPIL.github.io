"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _get_ap_indices():
    """Precompute all arithmetic progressions in [1, N] as a 2D array of indices."""
    aps = []
    for step in range(1, N // 2 + 1):
        for start in range(1, N - step + 1):
            # Determine max length for this start and step
            max_len = (N - start) // step + 1
            if max_len < 2:
                continue
            # For each start, step, we consider all lengths >= 2
            indices = [start - 1 + k * step for k in range(max_len)]
            for length in range(2, max_len + 1):
                aps.append(indices[:length])
    return aps

def _precompute_ap_matrix(aps):
    """
    Convert list of AP index lists into a structure that allows fast vectorized sum computation.
    We create a boolean mask or a sparse representation.
    Given N=256 and ~50k APs, a dense 50k x 256 matrix is ~10 million bools/ints, which is manageable.
    However, storing indices is better for memory, but for speed, we want to compute sums for ALL aps at once.
    
    Strategy:
    1. Create a list of all unique APs.
    2. For a given sign vector s (length N), we want to compute the sum for each AP.
    3. We can precompute a matrix M where M[i, j] = 1 if index j is in AP i, else 0.
    4. Then sums = M @ s.
    
    Let's check the size.
    Number of APs: 
    Step 1: ~128 starts * ~128 avg len / 2 ~ 8000
    Step 2: ~127 starts * ~64 avg len / 2 ~ 4000
    ...
    Total APs is roughly sum_{s=1}^{128} (256-s) * (256/s) / 2? No.
    For a fixed step s, start ranges 1..256-s. Max len is (256-start)/s + 1.
    Total APs is approximately O(N^2 log N). For N=256, it's around 50,000-100,000.
    A 100,000 x 256 float64 matrix is 100,000 * 256 * 8 bytes = ~200 MB. This is borderline but likely fits in memory.
    However, creating it every time is slow. We can create it once.
    """
    # We will store the indices in a list of arrays, but for the dot product, 
    # a dense matrix is fastest in NumPy.
    # Let's try to build the matrix.
    num_aps = len(aps)
    # To save memory and time, we can use int8 or float32 if we just need sums?
    # Sums are integers. But matrix @ vector needs consistent types.
    # Let's use a sparse approach if dense is too big? 
    # Actually, 50k * 256 = 12.8 million elements. float32 is 50MB. Very safe.
    
    # Initialize matrix with zeros
    # We need to be careful with the number of APs.
    # Let's count first.
    # The current _get_ap_indices returns a list.
    # We'll convert it to a numpy array of indices, then build the mask.
    
    # Alternative: Just store the list of index lists and use a loop?
    # No, the task is to minimize max abs sum. 
    # If we use a dense matrix M (num_aps x N), then for a sign vector s (N,), 
    # sums = M @ s. This is a single BLAS call, very fast.
    
    # Let's build M.
    M = np.zeros((num_aps, N), dtype=np.float32)
    for i, idxs in enumerate(aps):
        for idx in idxs:
            M[i, idx] = 1.0
            
    return M

def _compute_max_abs_sum_vectorized(signs, M):
    """Compute max abs sum using matrix multiplication."""
    # signs is a numpy array of length N
    # M is num_aps x N
    sums = M @ signs
    return np.max(np.abs(sums))

def _find_best_flip(signs, M, current_max):
    """
    Find the single flip that reduces the max abs sum the most.
    We can compute the new sums for all possible flips efficiently?
    If we flip index k, the new sum for AP i is old_sum_i + 2 * signs[k] * M[i, k].
    Let D_k = 2 * signs[k] * M[:, k]. This is a vector of length num_aps.
    New sums = sums + D_k.
    We want to minimize max(abs(sums + D_k)).
    
    Doing this for all k (N=256) means computing 256 vectors of length num_aps (~50k).
    256 * 50k = 12.8 million operations. This is very fast in NumPy if vectorized.
    
    Steps:
    1. Compute current sums: S = M @ signs (length num_aps)
    2. For each k in 0..N-1:
         delta = 2 * signs[k] * M[:, k]
         new_S = S + delta
         new_max = max(abs(new_S))
         if new_max < current_max:
             candidate = k
             best_max = new_max
    3. Return best k and best_max.
    """
    S = M @ signs
    best_k = -1
    best_max = current_max
    
    # Precompute M columns? No, M[:, k] is a view, which is fast.
    for k in range(N):
        # delta vector
        # signs[k] is scalar
        delta = 2.0 * signs[k] * M[:, k]
        new_S = S + delta
        new_max = np.max(np.abs(new_S))
        if new_max < best_max:
            best_max = new_max
            best_k = k
            
    return best_k, best_max

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit, keep 5s margin
    
    # 1. Precompute APs
    aps = _get_ap_indices()
    num_aps = len(aps)
    
    # 2. Precompute Matrix M
    # This might take a second or two.
    M = _precompute_ap_matrix(aps)
    
    # 3. Initial Candidates
    # Generate a few diverse starting points.
    # Random, Thue-Morse, etc.
    rng = random.Random(42)
    
    candidates = []
    
    # Random
    for i in range(5):
        signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.float32)
        candidates.append(signs)
        
    # Thue-Morse
    tm = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(N)], dtype=np.float32)
    candidates.append(tm)
    
    # Alternating
    alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.float32)
    candidates.append(alt)
    
    # Evaluate all to find best start
    best_signs = None
    best_val = float('inf')
    
    for cand in candidates:
        val = _compute_max_abs_sum_vectorized(cand, M)
        if val < best_val:
            best_val = val
            best_signs = cand.copy()
            
    # 4. Greedy Local Search (Exhaustive single flips)
    # Repeat until no improvement or time limit.
    # Since N=256, checking all 256 flips is cheap.
    # We can do many iterations.
    
    current_signs = best_signs
    current_max = best_val
    
    while time.time() < deadline - 2.0:
        best_k, new_max = _find_best_flip(current_signs, M, current_max)
        
        if best_k == -1:
            # No single flip improves the result. We are at a local minimum.
            break
            
        # Apply the flip
        current_signs[best_k] *= -1
        current_max = new_max
        
        # Optional: If we want to be more aggressive, we could try pairs, 
        # but single flips are usually enough to get to a good local min.
        # Given the speed of _find_best_flip (~1ms?), we can do thousands of iterations.
        
    # 5. Convert back to Python list of ints
    result = [int(x) for x in current_signs]
    
    return result

# EVOLVE-BLOCK-END
