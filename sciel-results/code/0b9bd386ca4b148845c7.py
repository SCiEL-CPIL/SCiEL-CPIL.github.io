"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _precompute_ap_data():
    """
    Precompute APs as a list of index arrays.
    Returns:
        ap_indices: List of numpy arrays, each containing 0-based indices of an AP.
        ap_lengths: List of lengths of each AP.
    """
    ap_indices = []
    ap_lengths = []
    for step in range(1, N // 2 + 1):
        for start in range(1, N - step + 1):
            max_len = (N - start) // step + 1
            if max_len < 2:
                continue
            # We consider all lengths >= 2
            # Note: The task says "including one", but singletons have sum +/- 1, 
            # so they are never the bottleneck for max_abs_sum > 1. 
            # However, to be strictly correct, we should include them or ensure 
            # our max is at least 1. Since we are minimizing max abs sum, 
            # and singletons are always 1, the global max is max(1, max_AP_sum).
            # We can ignore singletons in the optimization loop as long as we 
            # know the answer is >= 1.
            
            # Optimization: Instead of storing every length separately, 
            # we can just store the full AP and compute prefix sums?
            # No, the AP is a subsequence of the global array.
            # Storing explicit indices is memory heavy but N=256 is small.
            # Total number of APs:
            # Sum_{step=1}^{128} Sum_{start=1}^{256-step} (256-start)/step
            # This is roughly N^2 log N / 2 ?
            # Let's estimate: 
            # step=1: 255 starts, avg len ~128 -> ~32k APs? No.
            # For step=1, start=1: len 256. APs: [1..2], [1..3]... [1..256]. 255 APs.
            # start=2: len 255. APs: 254.
            # Total for step=1 is ~ N^2/2 = 32k.
            # For step=2: ~ N^2/8 = 8k.
            # Total ~ 32k * (1 + 1/2 + 1/4...) ~ 64k APs.
            # 64k APs * avg length ~ 10-20 indices. 
            # 64k * 20 * 4 bytes = 5 MB. Very manageable.
            
            indices = np.array([start - 1 + k * step for k in range(max_len)], dtype=np.int32)
            # We need to add all prefixes of length >= 2
            # Actually, we can just store the full array and use slicing during sum computation?
            # No, slicing creates copies or views that are hard to vectorize across different lengths.
            # Better to store the explicit index lists for each (start, step, length) tuple?
            # That creates ~64k arrays. 
            # Alternative: Store the full AP indices, and during evaluation, 
            # we need sum of signs[i] for i in AP[:L].
            
            # Let's stick to the original approach but optimize the storage.
            # Store the full index array.
            # When computing sums, we can use a trick:
            # If we have the full AP indices, the sum for length L is sum(signs[indices[:L]]).
            # This is hard to vectorize across all L for all APs simultaneously.
            
            # Let's go back to storing explicit index arrays for each valid AP.
            # It's the most straightforward for vectorized dot products if we flatten them.
            # But the lengths vary.
            
            # A better vectorized approach:
            # For each AP (start, step), we have a set of indices.
            # We can compute the prefix sums of signs along this AP.
            # But signs change.
            
            # Let's use the "Incidence Matrix" idea but sparse?
            # Or, simply, store the list of indices for each AP.
            # To evaluate all APs for a given sign vector:
            # We can loop over APs? No, that's slow.
            
            # Vectorized Evaluation Strategy:
            # 1. Precompute for each position j (0..255), the list of APs it belongs to?
            #    Too complex.
            
            # 2. Precompute the AP indices as a flat array and offsets (CSR format).
            #    Then, for a sign vector s, the sum of AP k is sum(s[indices[k]])
            #    This is a matrix-vector product where A is the incidence matrix.
            #    A is 64k x 256. 
            #    A @ s gives a vector of 64k sums.
            #    This is extremely fast with numpy.
            
            # Let's build the incidence matrix A (sparse or dense?).
            # 64k x 256 = 16 million entries. 
            # Dense float32: 16M * 4 bytes = 64 MB. 
            # Dense float64: 128 MB.
            # This is acceptable for a single run.
            
            pass # We will build the matrix in construct() to keep this function light or just return step/start info.
    
    # Actually, let's just return the list of (start, step, max_len) tuples to build the matrix later.
    # Or better, build the matrix here.
    
    # Count total APs first to pre-allocate
    total_aps = 0
    for step in range(1, N // 2 + 1):
        for start in range(1, N - step + 1):
            max_len = (N - start) // step + 1
            if max_len >= 2:
                total_aps += max_len - 1 # lengths 2..max_len
    
    # Create a dense matrix? 
    # 64,000 rows, 256 cols.
    # Let's use a sparse matrix from scipy? 
    # "Python standard library, NumPy and SciPy are available."
    # Sparse is better.
    
    from scipy.sparse import lil_matrix
    
    A = lil_matrix((total_aps, N), dtype=np.float32)
    
    row = 0
    for step in range(1, N // 2 + 1):
        for start in range(1, N - step + 1):
            max_len = (N - start) // step + 1
            if max_len < 2:
                continue
            indices = np.array([start - 1 + k * step for k in range(max_len)], dtype=np.int32)
            for length in range(2, max_len + 1):
                idxs = indices[:length]
                A[row, idxs] = 1.0
                row += 1
    
    return A.tocsr()

def _compute_max_abs_sum_vectorized(signs, A):
    """
    Compute max absolute sum over all APs.
    signs: numpy array of shape (N,)
    A: sparse matrix of shape (M, N)
    """
    sums = A.dot(signs)
    return np.max(np.abs(sums))

def _evaluate_all_flips(signs, A, current_sums):
    """
    For each position i, if we flip signs[i], what is the new max abs sum?
    
    Flipping signs[i] changes the sum of every AP containing i by -2 * signs[i].
    Let delta_i be the change in sum for AP k if we flip i.
    If i is in AP k: new_sum_k = current_sum_k - 2 * signs[i]
    Else: new_sum_k = current_sum_k
    
    We want to find the flip that minimizes max(abs(new_sums)).
    
    This is still hard to vectorize perfectly because the "max" operation is non-linear.
    However, we can approximate or use a heuristic.
    
    Heuristic: 
    The bottleneck APs are those with large |current_sum|.
    Flipping a sign that is in a bottleneck AP and has the same sign as the sum will reduce the sum magnitude.
    
    Let's identify the top K bottleneck APs.
    For each candidate flip i, compute the impact on these top K APs.
    Pick the flip that minimizes the max impact on these K APs.
    """
    N = len(signs)
    # Get the indices of the APs with the largest absolute sums
    abs_sums = np.abs(current_sums)
    # We don't need all, just the top few. 
    # Let's take the top 100 APs.
    k = min(100, len(current_sums))
    top_k_indices = np.argpartition(abs_sums, -k)[-k:]
    top_k_sums = current_sums[top_k_indices]
    
    # A_top is the submatrix for these APs (k x N)
    A_top = A[top_k_indices, :]
    
    # For each position i (0..N-1), we compute the new sums for the top K APs.
    # new_sums = top_k_sums - 2 * signs[i] * A_top[:, i]
    
    best_flip = -1
    best_max = float('inf')
    
    # Vectorize over positions?
    # A_top is k x N. 
    # signs is N.
    # We want for each i: max_j | top_k_sums[j] - 2 * signs[i] * A_top[j, i] |
    
    # Let's loop over i. N=256 is small.
    for i in range(N):
        # If we flip i, the change is -2 * signs[i] for APs where A_top[:, i] == 1
        # The new sums for the top K APs:
        new_sums = top_k_sums - 2.0 * signs[i] * A_top[:, i].toarray().flatten()
        new_max = np.max(np.abs(new_sums))
        
        # Also, we must consider that other APs (not in top K) might become the new max?
        # Unlikely if we are reducing the current max.
        # But to be safe, if new_max < current_max, we should verify with full computation?
        # That would be slow.
        # Let's assume the top K capture the bottleneck.
        
        if new_max < best_max:
            best_max = new_max
            best_flip = i
            
    return best_flip, best_max

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Precompute Incidence Matrix
    A = _precompute_ap_data()
    
    # 2. Initial Candidates
    rng = random.Random(42)
    candidates = []
    
    # Random
    for i in range(5):
        candidates.append(np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.float32))
    
    # Thue-Morse
    tm = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)], dtype=np.float32)
    candidates.append(tm)
    
    # Alternating
    alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)], dtype=np.float32)
    candidates.append(alt)
    
    # Best initial
    best_signs = None
    best_val = float('inf')
    best_sums = None
    
    for cand in candidates:
        sums = A.dot(cand)
        val = np.max(np.abs(sums))
        if val < best_val:
            best_val = val
            best_signs = cand.copy()
            best_sums = sums.copy()
            
    # 3. Local Search (Simulated Annealing style)
    current_signs = best_signs
    current_sums = best_sums
    current_max = best_val
    
    # Temperature for SA
    T = 1.0
    cooling = 0.95
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 2.0 and no_improve_count < max_no_improve:
        # Find the best flip using the top-K heuristic
        flip_idx, est_max = _evaluate_all_flips(current_signs, A, current_sums)
        
        if flip_idx == -1:
            break
            
        # Apply flip
        old_sign = current_signs[flip_idx]
        current_signs[flip_idx] = -old_sign
        
        # Update sums: sums -= 2 * old_sign * A[:, flip_idx]
        # A is sparse, so A[:, flip_idx] is a sparse vector.
        # We can do: current_sums -= 2.0 * old_sign * (A.getcol(flip_idx).toarray().flatten())
        # This is fast.
        col = A.getcol(flip_idx).toarray().flatten()
        current_sums -= 2.0 * old_sign * col
        
        new_max = np.max(np.abs(current_sums))
        
        # Acceptance
        delta = new_max - current_max
        if delta < 0:
            current_max = new_max
            no_improve_count = 0
            if new_max < best_val:
                best_val = new_max
                best_signs = current_signs.copy()
        else:
            # Accept with probability exp(-delta / T)
            # delta is positive (worse)
            # Since sums are integers, delta is at least 1 (or 2?)
            # Actually, flipping one sign changes sums by 2. So sums change by even integers.
            # Max abs sum changes by integer amounts.
            if T > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    # Accept worse solution
                    current_max = new_max
                    no_improve_count += 1
                else:
                    # Reject
                    current_signs[flip_idx] = old_sign
                    current_sums += 2.0 * old_sign * col
                    no_improve_count += 1
            else:
                # Greedy: reject
                current_signs[flip_idx] = old_sign
                current_sums += 2.0 * old_sign * col
                no_improve_count += 1
                
        T *= cooling
        
        # If we hit a very low value, we can stop or reduce T faster
        if current_max <= 2:
             break

    # Final Verification
    final_sums = A.dot(best_signs)
    final_val = np.max(np.abs(final_sums))
    
    # Convert to list of ints
    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
