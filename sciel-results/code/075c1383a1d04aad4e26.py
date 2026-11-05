"""Low discrepancy on all arithmetic progressions - local search approach."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list of +1/-1 indexed 0..n-1.
    Returns the max absolute sum.
    """
    n = len(signs)
    signs_arr = np.array(signs, dtype=np.int8)
    max_abs = 0
    
    for d in range(1, n + 1):
        # For each residue r in 0..d-1
        # We can process all residues for a fixed d by slicing
        # But lengths vary. 
        # Efficient way: 
        # For a fixed d, the APs are subsequences.
        # We can create a 2D array where rows are residue classes, padded with 0s to max length.
        # Then compute prefix sums along axis 1.
        # Then for each row, find max abs subarray sum.
        
        # Max length of any AP with step d is ceil(n/d).
        max_len = (n - 1) // d + 1
        
        # Create a padded array of shape (d, max_len)
        # Initialize with 0
        ap_matrix = np.zeros((d, max_len), dtype=np.int8)
        
        for r in range(d):
            # Indices: r, r+d, r+2d, ...
            # We can slice signs_arr[r::d]
            seq = signs_arr[r::d]
            length = len(seq)
            ap_matrix[r, :length] = seq
        
        # Compute prefix sums along axis 1
        # prefix[i, j] = sum(ap_matrix[i, 0:j+1])
        # We need prefix sums starting with 0? 
        # Standard Kadane's for max subarray sum on array A:
        # Let P be prefix sums of A, P[0]=0, P[k]=sum(A[0:k]).
        # Max subarray sum = max(P[j] - P[i]) for i<j.
        # This is equivalent to max(P) - min(P) is NOT correct.
        # It is max_{j} (P[j] - min_{i<j} P[i]).
        
        # Let's compute prefix sums P of shape (d, max_len + 1)
        # P[:, 0] = 0
        # P[:, k+1] = P[:, k] + ap_matrix[:, k]
        
        prefix = np.zeros((d, max_len + 1), dtype=np.int32)
        # Cumulative sum along axis 1
        # np.cumsum(ap_matrix, axis=1) gives sums ending at each index.
        # We need the running minimum of the prefix sums to compute max subarray sum.
        
        # Let S be the prefix sums of the AP sequences.
        # S[:, 0] = 0
        # S[:, k] = sum(ap_matrix[:, :k])
        
        # We can compute S using cumsum on ap_matrix, then prepend 0.
        # But ap_matrix has zeros in padding. 
        # We must be careful: the subarray must be within the valid length.
        # The valid length for row r is len(signs_arr[r::d]).
        # Let L[r] be the valid length.
        # We only care about subarrays within [0, L[r]].
        
        # Let's compute running min of prefix sums for each row, only up to valid length.
        # And running max of prefix sums.
        
        # Actually, we can just compute the max abs subarray sum for each row directly.
        
        # Let's do it row by row but vectorized where possible?
        # Or just use a loop over d, and inside, vectorize over r?
        
        # For fixed d, we have d rows.
        # We can compute the max abs subarray sum for all d rows at once?
        # The max abs subarray sum for a row is max(max_subarray, -min_subarray).
        
        # Let's compute P (prefix sums) for all rows.
        # P shape (d, max_len+1)
        # P[:, 0] = 0
        # P[:, 1:] = np.cumsum(ap_matrix, axis=1)
        
        P = np.zeros((d, max_len + 1), dtype=np.int32)
        if max_len > 0:
            P[:, 1:] = np.cumsum(ap_matrix, axis=1)
        
        # Now, for each row r, we want max_{0<=i<j<=L[r]} |P[r,j] - P[r,i]|
        # This is max( max_{j<=L[r]} (P[r,j] - min_{i<j, i<=L[r]} P[r,i]), 
        #              -min_{j<=L[r]} (P[r,j] - max_{i<j, i<=L[r]} P[r,i]) )
        
        # We can compute running min and max of P for each row, up to L[r].
        # However, L[r] varies. 
        # But note that for r < n % d (if n%d != 0), the length is floor(n/d)+1?
        # No, length is ceil((n-r)/d).
        # Let's just mask out the invalid parts.
        
        # Create a mask for valid indices in P.
        # Valid indices for row r are 0..L[r].
        # L[r] = len(signs_arr[r::d])
        lengths = np.array([len(signs_arr[r::d]) for r in range(d)], dtype=np.int32)
        
        # We need running min/max of P[:, :L[r]+1] for each row.
        # This is tricky with vectorization if lengths differ.
        # But d is at most 256. max_len is at most 256.
        # A simple loop over d is fine. Inside, we can do vectorized ops on the d rows.
        
        # For fixed d, P is (d, max_len+1).
        # We want to compute max_abs_subarray for each of the d rows.
        
        # Let's compute running min of P along axis 1.
        # But we must ignore values beyond L[r].
        # We can set P[r, L[r]+1:] to a very large number for min calculation?
        # And very small for max calculation?
        
        # For Max Subarray Sum:
        # We need min_{i<j} P[i]. 
        # If we set P[r, k] = +inf for k > L[r], then the running min will be correct for j <= L[r].
        # But we also need to ensure we don't pick i > L[r].
        # If we set P[r, k] = +inf for k > L[r], then min_{i<=j} P[i] for j<=L[r] is unaffected by i>L[r] because i<=j.
        # So we can just mask the tail.
        
        P_masked_min = P.copy()
        # For each row r, set indices > lengths[r] to +inf
        # This is a bit slow to do in a loop.
        
        # Alternative: Since n=256 is small, maybe the pure Python version with optimizations is enough?
        # The previous pure Python version was ~16M ops. 
        # A NumPy version should be < 1ms per evaluation.
        
        # Let's try a simpler NumPy approach:
        # Just loop over d.
        # Inside, loop over r? No, vectorize over r.
        
        # Let's use the fact that we can compute max subarray sum using:
        # max_subarray = max(P) - min(P) is wrong.
        # 
        # Let's stick to a clean vectorized implementation for fixed d.
        
        # 1. Create P (d, max_len+1)
        # 2. Mask invalid parts of P for min/max calculations.
        #    For calculating max(P[j] - min(P[i])):
        #    We need min(P[i]) for i < j.
        #    If we compute running_min = np.minimum.accumulate(P, axis=1), 
        #    it will include invalid parts if they are small.
        #    So we must set invalid parts to +inf.
        
        # Let's create a mask of valid indices.
        # indices = np.arange(max_len + 1)
        # valid_mask = indices[None, :] <= lengths[:, None]
        
        # P_for_min = np.where(valid_mask, P, np.inf)
        # running_min = np.minimum.accumulate(P_for_min, axis=1)
        # max_diff = P - running_min
        # max_subarray = np.max(max_diff)
        
        # Similarly for min subarray:
        # P_for_max = np.where(valid_mask, P, -np.inf)
        # running_max = np.maximum.accumulate(P_for_max, axis=1)
        # min_diff = P - running_max
        # min_subarray = np.min(min_diff)
        
        # max_abs = max(max_subarray, -min_subarray)
        
        indices = np.arange(max_len + 1)
        valid_mask = indices[None, :] <= lengths[:, None]
        
        P_for_min = np.where(valid_mask, P, np.inf)
        running_min = np.minimum.accumulate(P_for_min, axis=1)
        max_diff = P - running_min
        # We only care about valid j.
        # max_diff for invalid j might be large, but P is 0 there?
        # Wait, P is 0 for invalid parts if we initialized with 0?
        # No, P[:, 1:] = cumsum(ap_matrix). ap_matrix is 0 for invalid parts.
        # So P is constant for invalid parts.
        # But we masked P_for_min with inf.
        # So running_min is correct for valid j.
        # For invalid j, max_diff = P - running_min.
        # We should mask max_diff to only consider valid j.
        max_diff_masked = np.where(valid_mask, max_diff, -np.inf)
        max_subarray_val = np.max(max_diff_masked)
        
        P_for_max = np.where(valid_mask, P, -np.inf)
        running_max = np.maximum.accumulate(P_for_max, axis=1)
        min_diff = P - running_max
        min_diff_masked = np.where(valid_mask, min_diff, np.inf)
        min_subarray_val = np.min(min_diff_masked)
        
        current_max_abs = max(max_subarray_val, -min_subarray_val)
        if current_max_abs > max_abs:
            max_abs = current_max_abs
            
    return int(max_abs)

def construct():
    deadline = time.time() + 50  # 10s safety margin
    n = 256
    rng = random.Random(42)
    
    # Initialize with random signs
    best_signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_val = _max_ap_discrepancy(best_signs)
    
    # Local search: try flipping individual signs
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n):
            old = best_signs[i]
            best_signs[i] = -old
            val = _max_ap_discrepancy(best_signs)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_signs[i] = old
    
    # Try multiple random restarts with local search
    while time.time() < deadline - 10:
        signs = [rng.choice((-1, 1)) for _ in range(n)]
        val = _max_ap_discrepancy(signs)
        if val < best_val:
            best_signs = signs[:]
            best_val = val
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                old = signs[i]
                signs[i] = -old
                v = _max_ap_discrepancy(signs)
                if v < val:
                    val = v
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_signs = signs[:]
                else:
                    signs[i] = old
    
    return best_signs
# EVOLVE-BLOCK-END
