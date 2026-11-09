import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Finds 30 integer coefficients from {-1, 0, 1} such that the signed sum
    sum(c[i-1]/i) is non-zero and its absolute value is minimized.
    
    Strategy:
    1. Meet-in-the-Middle: Split the 30 coefficients into two sets of 15.
    2. Enumerate all 3^15 = 14,348,907 possible signed sums for each half.
       Use exact rational arithmetic via integer numerators and denominators
       (LCM of 1..30) to avoid floating point errors and Fraction overhead.
    3. Sort the sums for the second half.
    4. For each sum in the first half, use binary search to find the sum in the
       second half that brings the total sum closest to zero.
    5. Track the best non-zero total sum.
    6. Reconstruct the coefficients from the best pair of indices.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 30
    half_n = 15
    
    # Precompute LCM of 1..30 for exact integer representation of fractions
    # LCM(1..30) = 2329089562800
    # We can store sums as integers: sum(c_i / i) = sum(c_i * (LCM / i)) / LCM
    # Let L = LCM(1..30).
    # weight[i] = L // i
    # sum_val = sum(c[i-1] * weight[i-1])
    # We want to minimize |sum_val1 + sum_val2| != 0
    
    # Calculate LCM
    import math
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // math.gcd(lcm_val, i)
        
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    # Split weights into two halves
    w1 = weights[:half_n]
    w2 = weights[half_n:]
    
    # Function to generate all sums for a given set of weights
    # Returns: array of sums (int64), and mapping from sum_index to coefficients
    # Actually, storing mapping for 14M items is heavy. 
    # We can reconstruct coefficients from the index in base-3 if we generate them in order.
    # Or we can just store the sums, find the best pair, and then decode the indices.
    
    def generate_sums(weights_list):
        """
        Generates all 3^len(weights_list) sums.
        Returns a numpy array of int64 sums.
        The index in the array corresponds to the base-3 representation of the coefficients.
        Index 0: all 0s
        Index 1: last coeff is 1 (if we map 0->-1, 1->0, 2->1? No, standard base 3)
        Let's map:
        digit 0 -> coeff -1
        digit 1 -> coeff 0
        digit 2 -> coeff 1
        
        Wait, standard base 3 expansion:
        index = d_0 * 3^0 + d_1 * 3^1 + ...
        coeff[k] = d_k - 1
        
        We need to generate sums efficiently.
        """
        m = len(weights_list)
        num_sums = 3 ** m
        sums = np.zeros(num_sums, dtype=np.int64)
        
        # Dynamic programming / Iterative construction
        # Start with [0]
        # For each weight, new_sums = [s - w, s, s + w for s in old_sums]
        # This is 3x expansion each step.
        
        current_sums = np.array([0], dtype=np.int64)
        
        for w in weights_list:
            # Expand
            # We need to interleave or concatenate.
            # new_sums = np.concatenate([current_sums - w, current_sums, current_sums + w])
            # This creates the order where the new digit is the most significant?
            # Let's check index mapping.
            # If we do concat([A-w, A, A+w]), the first 1/3 are A-w, next 1/3 are A, last 1/3 are A+w.
            # This corresponds to the new coefficient being the "last" one in the base-3 logic if we think of it recursively.
            # Let's just ensure we can decode the index later.
            
            # Using np.repeat and np.tile might be faster or slower depending on size.
            # Let's use simple concatenation for clarity, it's fast enough for 14M.
            # However, 14M * 8 bytes = 112 MB per array. 3 arrays = 336 MB.
            # This fits in memory.
            
            current_sums = np.concatenate([
                current_sums - w,
                current_sums,
                current_sums + w
            ])
            
            if time.time() > deadline:
                # Fallback to trivial if generation takes too long (unlikely)
                break
                
        return current_sums

    # Generate sums for first half
    sums1 = generate_sums(w1)
    
    # Generate sums for second half
    sums2 = generate_sums(w2)
    
    # Sort sums2 and keep track of original indices
    # We need to find pairs (s1, s2) such that |s1 + s2| is minimized and non-zero.
    # Sort sums2.
    sorted_indices2 = np.argsort(sums2)
    sorted_sums2 = sums2[sorted_indices2]
    
    best_val = None
    best_idx1 = -1
    best_idx2 = -1
    
    # Iterate over sums1
    # For each s1, we want s2 such that s1 + s2 is close to 0.
    # So s2 should be close to -s1.
    
    # Optimization: We can vectorize the binary search part?
    # np.searchsorted(sorted_sums2, -sums1)
    
    targets = -sums1
    # Find insertion points
    # We check the element at pos and pos-1
    
    pos = np.searchsorted(sorted_sums2, targets)
    
    # Clip pos to valid range [0, len(sums2)-1] for direct access
    # We need to check pos and pos-1
    
    # To avoid array of size 14M for candidates, we can process in chunks or just vectorize the check.
    # Let's vectorize.
    
    # Candidate 1: sorted_sums2[pos]
    # Candidate 2: sorted_sums2[pos-1]
    
    # Be careful with boundaries
    pos_clipped = np.clip(pos, 0, len(sums2) - 1)
    
    cand1 = sorted_sums2[pos_clipped]
    cand2 = np.where(pos > 0, sorted_sums2[pos - 1], -np.inf)
    
    total1 = sums1 + cand1
    total2 = sums1 + cand2
    
    abs_total1 = np.abs(total1)
    abs_total2 = np.abs(total2)
    
    # We want min(abs_total) where abs_total > 0
    # Mask out zeros
    mask1 = abs_total1 > 0
    mask2 = abs_total2 > 0
    
    # Set invalid ones to inf
    abs_total1_valid = np.where(mask1, abs_total1, np.inf)
    abs_total2_valid = np.where(mask2, abs_total2, np.inf)
    
    # Combine
    final_abs = np.minimum(abs_total1_valid, abs_total2_valid)
    
    # Find global minimum
    min_abs = np.min(final_abs)
    
    if min_abs == np.inf:
        # Should not happen, but fallback
        return [0] * 28 + [1, -1]
        
    # Find the index in sums1 that gave this minimum
    min_idx1 = np.argmin(final_abs)
    
    # Determine which candidate (1 or 2) was the winner
    if abs_total1_valid[min_idx1] <= abs_total2_valid[min_idx1]:
        # Winner is candidate 1
        winner_pos = pos_clipped[min_idx1]
        # The index in the original sums2 array is sorted_indices2[winner_pos]
        orig_idx2 = sorted_indices2[winner_pos]
    else:
        # Winner is candidate 2
        # pos[min_idx1] is the insertion point. Candidate 2 is at pos-1.
        p = pos[min_idx1]
        winner_pos = p - 1
        orig_idx2 = sorted_indices2[winner_pos]
        
    # Now we have min_idx1 (index in sums1) and orig_idx2 (index in sums2)
    # We need to decode these indices into coefficients.
    
    def decode_index(idx, m):
        """
        Decodes a base-3 index into m coefficients in {-1, 0, 1}.
        The generation order was:
        Start: [0]
        Step 1: [0-w0, 0, 0+w0] -> indices 0, 1, 2 map to coeffs [-1, 0, 1] for weight w0
        Step 2: Expand each.
        The index structure is essentially base-3 where the least significant digit corresponds to the FIRST weight processed (w0), 
        and the most significant digit corresponds to the LAST weight processed (w_{m-1}).
        
        Let's verify:
        m=1. idx 0 -> -1*w0. idx 1 -> 0. idx 2 -> 1*w0.
        Base 3: 0 -> 0 (-1), 1 -> 1 (0), 2 -> 2 (1). Correct.
        
        m=2.
        Start [0]
        After w0: [ -w0, 0, w0 ] (indices 0, 1, 2)
        After w1:
        idx 0: (-w0) - w1. Base 3: 00. Digits: 0,0. Coeffs: -1, -1. Sum: -w0 - w1. Correct.
        idx 1: (-w0) + 0.  Base 3: 01. Digits: 1,0. Coeffs: 0, -1. Sum: 0 - w1. Correct.
        idx 2: (-w0) + w1. Base 3: 02. Digits: 2,0. Coeffs: 1, -1. Sum: w0 - w1. Correct.
        idx 3: 0 - w1.     Base 3: 10. Digits: 0,1. Coeffs: -1, 0. Sum: -w0. Correct.
        
        So, digit k (from right, 0-indexed) corresponds to weight w_k.
        digit = (idx // 3^k) % 3
        coeff = digit - 1
        """
        coeffs = [0] * m
        for k in range(m):
            digit = (idx // (3 ** k)) % 3
            coeffs[k] = digit - 1
        return coeffs
    
    coeffs1 = decode_index(min_idx1, half_n)
    coeffs2 = decode_index(orig_idx2, half_n)
    
    all_coeffs = coeffs1 + coeffs2
    
    # Verify
    s = 0
    for c, i in zip(all_coeffs, range(1, n+1)):
        if c != 0:
            s += c * (lcm_val // i)
            
    if s == 0:
        # Fallback if something went wrong (should be prevented by mask)
        return [0] * 28 + [1, -1]
        
    return all_coeffs

# EVOLVE-BLOCK-END
