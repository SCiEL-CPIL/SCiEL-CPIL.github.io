"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    if limit < 1:
        return [1]
    powerful = set()
    powerful.add(1)
    
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def _is_powerful(n):
    """Check if n is powerful using exact factorization."""
    if n <= 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count == 1:
                return False
        p += 1
    if temp > 1:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1
    
    # Try increasing limits
    limits_to_try = [10**8, 2*10**8]
    
    for limit in limits_to_try:
        elapsed = time.time() - start_time
        if elapsed > deadline - 35:
            break
        
        # Generate powerful numbers up to limit
        powerful = _generate_powerful(limit)
        powerful_set = set(powerful)
        arr = np.array(powerful, dtype=np.int64)
        M = len(arr)
        
        if M < 2:
            continue
        
        # Limit M to keep computation manageable
        # The number of powerful numbers up to 1e8 is around 10000+
        # We need to be careful with M^2 operations
        
        # Use chunked approach to collect all valid sums
        # For each i, find j range where i <= j and arr[i] + arr[j] <= limit
        
        # Precompute j_max for each i
        max_vals = limit - arr
        j_max_all = np.searchsorted(arr, max_vals, side='right') - 1
        
        # For each i, valid j are from i to j_max_all[i]
        # We want to collect all sums arr[i] + arr[j] for valid (i,j)
        
        # To avoid O(M^2) memory, process in chunks
        all_sums_parts = []
        chunk_size = 100  # Process 100 i values at a time
        
        for i_start in range(0, M, chunk_size):
            i_end = min(i_start + chunk_size, M)
            chunk_sums = []
            
            for i in range(i_start, i_end):
                ai = arr[i]
                j_max = j_max_all[i]
                if j_max < i:
                    continue
                # Collect sums for j in [i, j_max]
                sums = ai + arr[i:j_max+1]
                chunk_sums.append(sums)
            
            if chunk_sums:
                all_sums_parts.append(np.concatenate(chunk_sums))
            
            if time.time() - start_time > deadline - 25:
                break
        
        if time.time() - start_time > deadline - 25:
            break
        
        if not all_sums_parts:
            continue
        
        # Concatenate all sums
        flat_sums = np.concatenate(all_sums_parts)
        
        # Filter to valid range [2, limit]
        flat_sums = flat_sums[(flat_sums >= 2) & (flat_sums <= limit)]
        
        if len(flat_sums) == 0:
            continue
        
        # Count occurrences of each sum
        # Use np.unique with return_counts
        unique_sums, counts_arr = np.unique(flat_sums, return_counts=True)
        
        # Find the sum with maximum count
        best_idx = int(np.argmax(counts_arr))
        best_N_here = int(unique_sums[best_idx])
        best_count_here = int(counts_arr[best_idx])
        
        if best_count_here > best_count:
            best_count = best_count_here
            # Extract the pairs for this N
            pairs = []
            for a in powerful:
                if 2 * a > best_N_here:
                    break
                b = best_N_here - a
                if b in powerful_set and a <= b:
                    pairs.append([a, b])
            
            # Verify count matches
            if len(pairs) == best_count_here:
                best_result = {'N': best_N_here, 'pairs': pairs}
            elif len(pairs) > best_count:
                best_count = len(pairs)
                best_result = {'N': best_N_here, 'pairs': pairs}
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
    
    return best_result

# EVOLVE-BLOCK-END
