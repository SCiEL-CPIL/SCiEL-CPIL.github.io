import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    # Generate powerful numbers up to a limit M
    # A powerful number n has all prime factors with exponent >= 2
    # So n = a^2 * b^3 where b is square-free
    # Count of powerfuls up to X is ~ c * X^(2/5)
    
    # We want a large M to get many powerful numbers
    # For M = 10^9, K ~ (10^9)^0.4 = 10^3.6 ≈ 4000
    # For M = 10^10, K ~ (10^10)^0.4 = 10^4 = 10000
    # O(K^2) with K=10000 is 100M operations, too slow in pure Python
    # But with NumPy vectorization, we can process chunks
    
    M = 1_000_000_000  # 10^9
    
    # Precompute square-free check using sieve
    max_b = int(M ** (1.0 / 3.0)) + 2  # ~1000 for M=10^9
    
    # Sieve for square-free numbers up to max_b
    is_sf = np.ones(max_b + 1, dtype=bool)
    is_sf[0] = False
    is_sf[1] = True
    for p in range(2, int(np.sqrt(max_b)) + 1):
        if is_sf[p]:
            p2 = p * p
            is_sf[p2::p2] = False
    
    powerful_list = []
    
    for b in range(1, max_b + 1):
        if not is_sf[b]:
            continue
        b_cubed = b ** 3
        if b_cubed > M:
            break
        max_a = isqrt(M // b_cubed)
        # Generate all a^2 * b_cubed for a in 1..max_a
        a_vals = np.arange(1, max_a + 1, dtype=np.int64)
        vals = (a_vals * a_vals) * b_cubed
        powerful_list.extend(vals.tolist())
    
    if time.time() > deadline - 30:
        # Fallback: use a smaller M
        M = 100_000_000
        max_b = int(M ** (1.0 / 3.0)) + 2
        is_sf = np.ones(max_b + 1, dtype=bool)
        is_sf[0] = False
        is_sf[1] = True
        for p in range(2, int(np.sqrt(max_b)) + 1):
            if is_sf[p]:
                p2 = p * p
                is_sf[p2::p2] = False
        powerful_list = []
        for b in range(1, max_b + 1):
            if not is_sf[b]:
                continue
            b_cubed = b ** 3
            if b_cubed > M:
                break
            max_a = isqrt(M // b_cubed)
            a_vals = np.arange(1, max_a + 1, dtype=np.int64)
            vals = (a_vals * a_vals) * b_cubed
            powerful_list.extend(vals.tolist())
    
    # Remove duplicates and sort
    powerful_set = set(powerful_list)
    powerful_arr = np.array(sorted(powerful_set), dtype=np.int64)
    K = len(powerful_arr)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Vectorized O(K^2) counting using NumPy
    # Process in chunks to avoid memory issues
    # For each chunk of 'a' values, compute all sums with 'b' values (b >= a)
    
    best_N = 0
    best_count = 0
    counts = {}
    
    # Use a chunked approach
    chunk_size = 500
    for i_start in range(0, K, chunk_size):
        i_end = min(i_start + chunk_size, K)
        a_chunk = powerful_arr[i_start:i_end]
        
        # For each a in a_chunk, we need b >= a
        # We can compute all sums for this chunk
        # sums[i, j] = a_chunk[i] + powerful_arr[j] for j >= i_start
        
        # To handle b >= a, we need to be careful
        # Let's just compute all sums and filter later, or use masking
        
        # Create the full sum matrix for this chunk
        # This could be large, so let's do it in sub-chunks if needed
        sub_chunk_size = 100
        for si_start in range(0, len(a_chunk), sub_chunk_size):
            si_end = min(si_start + sub_chunk_size, len(a_chunk))
            a_sub = a_chunk[si_start:si_end]
            
            # For each a in a_sub, find valid b indices
            # b must be >= a and in powerful_arr
            # Since powerful_arr is sorted, for a value, b starts from searchsorted(a)
            
            for idx, a_val in enumerate(a_sub):
                # Find start index for b >= a_val
                b_start = np.searchsorted(powerful_arr, a_val)
                if b_start >= K:
                    continue
                
                b_vals = powerful_arr[b_start:]
                sums = a_val + b_vals
                
                # Count these sums
                for s in sums:
                    s_int = int(s)
                    counts[s_int] = counts.get(s_int, 0) + 1
                    if counts[s_int] > best_count:
                        best_count = counts[s_int]
                        best_N = s_int
            
            # Time check
            if time.time() > deadline - 15:
                break
        
        if time.time() > deadline - 15:
            break
    
    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Reconstruct pairs for best_N
    pairs = []
    limit = best_N // 2
    end_idx = np.searchsorted(powerful_arr, limit)
    
    for i in range(end_idx):
        a = int(powerful_arr[i])
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
    
    return {'N': int(best_N), 'pairs': pairs}
