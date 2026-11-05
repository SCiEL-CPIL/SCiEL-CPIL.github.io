"""Low discrepancy on all arithmetic progressions - Local search with vectorized evaluation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or array of +1/-1 indexed 0..n-1.
    Returns the max absolute sum.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    
    # We iterate over step sizes d.
    # For a given d, the progressions are determined by start index i (0 <= i < d)
    # and length L.
    # An AP starting at i with step d is s[i], s[i+d], s[i+2d], ...
    # We want max |sum_{k=0}^{L-1} s[i + k*d]| for all valid i, L.
    
    # Optimization:
    # For a fixed d, we can process all starting indices i in [0, d) simultaneously.
    # Let's create a matrix M of shape (d, max_len) where M[i, k] = s[i + k*d] if valid, else 0.
    # Then compute cumulative sums along axis 1.
    # The prefix sums P[i, j] = sum_{k=0}^{j-1} M[i, k].
    # We need max |P[i, j] - P[i, k]| for 0 <= k < j <= max_len_i.
    # This is equivalent to max(P[i, :]) - min(P[i, :])? No.
    # It is max_{j} (P[i, j] - min_{k<j} P[i, k]) and -min_{j} (P[i, j] - max_{k<j} P[i, k]).
    
    # A simpler approach for small n=256:
    # Iterate d from 1 to n.
    # For each d, create the matrix of residues.
    # Compute prefix sums.
    # For each row, compute max subarray sum and min subarray sum.
    
    for d in range(1, n + 1):
        max_len = (n - 1) // d + 1
        # Initialize matrix with zeros
        # Shape (d, max_len)
        mat = np.zeros((d, max_len), dtype=np.int16)
        
        # Fill matrix
        # For r in 0..d-1:
        #   indices = r, r+d, ...
        #   mat[r, :] = s[r::d]
        
        # Vectorized fill is tricky due to varying lengths, but n is small.
        # Let's just use slicing in a loop for d, which is O(n^2) total operations roughly.
        # Actually, filling takes sum_{d=1}^n d * (n/d) = n^2 operations.
        # n=256, n^2 = 65536. This is very fast in Python if not inside a tight loop?
        # No, Python loops are slow. Let's try to vectorize the fill or use a different strategy.
        
        # Alternative: Just compute the sums directly?
        # For each d, for each r, we have a sequence.
        # We can compute the prefix sums of the sequence.
        
        # Let's stick to the matrix approach but optimize the filling.
        # We can construct the indices:
        # idx = np.arange(n)
        # r = idx % d
        # k = idx // d
        # mat[r, k] = s[idx]
        
        idx = np.arange(n)
        r = idx % d
        k = idx // d
        mat[r, k] = s[idx]
        
        # Now mat has shape (d, max_len).
        # Note: Some entries in the last column might be from different residues?
        # No, k = idx // d.
        # For a fixed d, max_len is the max k+1.
        # The matrix mat is (d, max_len).
        # Entries where k >= len(sequence for r) are 0 because initialized to 0 and not overwritten.
        # This is correct because we want to treat missing elements as 0 for prefix sum continuity?
        # Wait. If we pad with 0, the prefix sum continues.
        # Example: s = [1, -1, 1], d=2.
        # r=0: indices 0, 2 -> s[0]=1, s[2]=1. seq: [1, 1].
        # r=1: indices 1 -> s[1]=-1. seq: [-1].
        # mat[0] = [1, 1] (padded to max_len=2)
        # mat[1] = [-1, 0] (padded to max_len=2)
        
        # Prefix sums P = np.cumsum(mat, axis=1)
        # P[0] = [1, 2]
        # P[1] = [-1, -1]
        
        # We need max |P[i, j] - P[i, k]| for valid k, j.
        # Valid k, j are those where the index in the original array is < n.
        # In the matrix, valid k for row i is 0..len(seq_i)-1.
        # The prefix sum P[i, j] represents sum of first j elements of the padded sequence.
        # If we include the padded 0s, the sum doesn't change.
        # So max |sum| over any subarray of the valid sequence is the same as
        # max |sum| over any subarray of the padded sequence?
        # Yes, because adding 0s at the end doesn't create new non-zero sums,
        # but it extends the range of j.
        # However, we must ensure we don't pick a subarray that starts in the valid part
        # and ends in the padded part in a way that implies a sum that didn't exist?
        # No, sum(a..b) where b is in padded part is sum(a..end_valid).
        # So it's fine.
        
        P = np.cumsum(mat, axis=1)
        
        # We want max |P[i, j] - P[i, k]|.
        # This is max_{i} (max_j P[i,j] - min_{k<j} P[i,k])?
        # Standard algorithm:
        # For each row i:
        #   min_prefix = 0
        #   max_sub = 0
        #   for j in range(max_len):
        #       min_prefix = min(min_prefix, P[i, j-1]) # Careful with indexing
        #       ...
        
        # Vectorized:
        # Let's compute the running min of P.
        # running_min = np.minimum.accumulate(P, axis=1)
        # This gives min(P[i, 0..j]) for each j.
        # Then P[i, j] - running_min[i, j] is the max sum ending at j?
        # No, max sum ending at j is P[i, j] - min(P[i, 0..j-1]).
        # np.minimum.accumulate includes the current element.
        # So we need min(P[i, 0..j-1]).
        
        # Let's shift P by 1?
        # Or just compute:
        # min_prefix_before = np.zeros_like(P)
        # min_prefix_before[:, 0] = 0
        # min_prefix_before[:, 1:] = np.minimum.accumulate(P[:, :-1], axis=1)
        
        # This is getting complex to vectorize perfectly without loops over d.
        # Given n=256, a loop over d (256 iterations) and inside a vectorized op on arrays of size <= 256x256 is fast.
        
        # Let's try a simpler, slightly slower but robust vectorization for each d.
        
        # P shape (d, max_len)
        # We want max_abs_subarray for each row.
        
        # Max subarray sum:
        # Let's use the property: max_subarray = max(P) - min(P) is WRONG.
        
        # Let's just loop over the d rows? No, d is up to 256.
        # Let's use a vectorized trick for max subarray sum.
        
        # Actually, for n=256, even a pure Python loop over d and r might be acceptable if optimized?
        # 256 * 256 = 65k iterations. Inside, we do O(1) work if we precomputed?
        # No, inside we need to find max subarray sum of a sequence of length L.
        # That is O(L).
        # Total O(n^2). 65k operations. This is < 1ms in C, but in Python?
        # 65k Python iterations is ~10-50ms.
        # Plus the numpy overhead.
        
        # Let's try to keep it vectorized as much as possible.
        
        # Compute prefix sums P (d, max_len)
        # We need to compute max(P[i, j] - min_{k<j} P[i, k]) and -min(P[i, j] - max_{k<j} P[i, k])
        
        # Let's create P_shifted where P_shifted[i, j] = min(P[i, 0..j-1])
        # This is hard to do directly with np.minimum.accumulate because it includes current.
        
        # Alternative:
        # max_subarray_sum = np.max(P - np.minimum.accumulate(P, axis=1)[:, :-1] - P[:, :-1]?) No.
        
        # Let's just do the loop over d, and inside use numpy for the row operations.
        # For fixed d, P is (d, max_len).
        # We can compute max subarray sum for all rows at once?
        
        # Let's try:
        # min_prefix = np.zeros_like(P)
        # min_prefix[:, 0] = 0
        # min_prefix[:, 1:] = np.minimum.accumulate(P[:, :-1], axis=1)
        
        # max_prefix = np.zeros_like(P)
        # max_prefix[:, 0] = 0
        # max_prefix[:, 1:] = np.maximum.accumulate(P[:, :-1], axis=1)
        
        # max_diff = P - min_prefix
        # min_diff = P - max_prefix
        
        # max_sub = np.max(max_diff)
        # min_sub = np.min(min_diff)
        
        # max_abs = max(max_sub, -min_sub)
        
        # This should work and be vectorized over the d rows.
        
        min_prefix = np.zeros_like(P)
        max_prefix = np.zeros_like(P)
        
        if max_len > 1:
            min_prefix[:, 1:] = np.minimum.accumulate(P[:, :-1], axis=1)
            max_prefix[:, 1:] = np.maximum.accumulate(P[:, :-1], axis=1)
        
        max_diff = P - min_prefix
        min_diff = P - max_prefix
        
        # Note: We must ensure we only consider valid parts of the sequence.
        # The padding with 0s in `mat` means P continues flat.
        # Does this affect the max/min?
        # If the sequence ends, P stays constant.
        # So max_diff might be 0 for the tail.
        # This is safe.
        
        local_max = max(np.max(max_diff), -np.min(min_diff))
        if local_max > max_abs:
            max_abs = local_max
            
    return int(max_abs)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = 256
    rng = random.Random(42)
    
    # Initial candidates
    # 1. Random
    # 2. Walsh-Hadamard (Thue-Morse like)
    # 3. Van der Corput
    
    def thue_morse(n):
        # t(i) = popcount(i) % 2
        # Map 0->1, 1->-1
        return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    
    def van_der_corput(n):
        # Not a standard +/-1 sequence, need to adapt or just use random.
        # Let's use a simple alternating pattern or random.
        return [rng.choice((-1, 1)) for _ in range(n)]
        
    candidates = []
    candidates.append([rng.choice((-1, 1)) for _ in range(n)])
    candidates.append(thue_morse(n))
    candidates.append([1 if i % 2 == 0 else -1 for i in range(n)])
    
    # Evaluate initial candidates
    best_signs = None
    best_score = float('inf')
    
    for c in candidates:
        score = _max_ap_discrepancy(c)
        if score < best_score:
            best_score = score
            best_signs = c[:]
            
    # Local Search
    # Single bit flip
    # We can try to flip a bit that reduces the max discrepancy.
    # To speed up, we can track which APs are "tight" (achieving the max).
    # But for simplicity, let's just do random flips and evaluate.
    # With vectorized eval, 1 eval is ~1ms?
    # 85s / 1ms = 85000 evals.
    # This should be enough for good improvement.
    
    current_signs = best_signs[:]
    current_score = best_score
    
    # Simulated Annealing or Hill Climbing with restarts?
    # Let's do Hill Climbing with random restarts on plateau.
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = rng.randint(0, n - 1)
        old_val = current_signs[idx]
        current_signs[idx] = -old_val
        
        new_score = _max_ap_discrepancy(current_signs)
        
        if new_score < current_score:
            current_score = new_score
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_signs = current_signs[:]
        else:
            # Revert if not improved (Hill Climbing)
            # Or accept with probability for SA?
            # Let's stick to strict hill climbing for now, with restarts.
            current_signs[idx] = old_val
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Restart from best or random
            if rng.random() < 0.5:
                current_signs = best_signs[:]
                current_score = best_score
            else:
                current_signs = [rng.choice((-1, 1)) for _ in range(n)]
                current_score = _max_ap_discrepancy(current_signs)
                if current_score < best_score:
                    best_score = current_score
                    best_signs = current_signs[:]
            no_improve_count = 0
            
    return best_signs

# EVOLVE-BLOCK-END
