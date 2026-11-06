import time
import numpy as np

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to compute the small-prime part (23-smooth part) of each number
    # in the range [1001, 1000024] (i.e., n+1 to n+24 for n in [1000, 1000000])
    
    # Total range: 1000024 - 1001 + 1 = 1000024 numbers
    # Let's index from 0: arr[i] corresponds to value 1001 + i
    start_val = 1001
    end_val = 1000024
    total = end_val - start_val + 1  # 1000024
    
    # Compute the small-prime part for each number in the range
    # small_part[m] = product of p^v_p(m) for p in primes
    # We'll compute this using a segmented approach
    
    # Initialize with 1
    small_parts = np.ones(total, dtype=np.int64)
    
    # For each prime, divide out all factors
    # For efficiency, we process each prime by finding multiples in the range
    for p in primes:
        # Find the first multiple of p in [start_val, end_val]
        first = ((start_val + p - 1) // p) * p
        for m in range(first, end_val + 1, p):
            idx = m - start_val
            val = small_parts[idx]
            while val % p == 0:
                val //= p
            # Wait, this is wrong. We need to extract p^v_p from the original number,
            # not from the already-modified small_parts.
            # Let me reconsider.
    
    # Actually, let's compute it differently.
    # small_part[m] should be the product of all prime powers p^a where p < 24 and p^a || m
    # i.e., the largest divisor of m composed only of primes < 24
    
    # Better approach: for each number in range, factor out small primes
    # But doing this for 1M numbers individually is slow in Python.
    
    # Let's use a sieve-like approach:
    # Initialize small_parts to 1
    # For each prime p, for each multiple of p in the range, multiply by p for each power
    
    small_parts = np.ones(total, dtype=np.int64)
    
    for p in primes:
        # For each power p^k, mark multiples
        pk = p
        while pk <= end_val:
            first = ((start_val + pk - 1) // pk) * pk
            for m in range(first, end_val + 1, pk):
                idx = m - start_val
                small_parts[idx] *= p
            pk *= p
    
    # Now small_parts[i] is the small-prime part of (start_val + i)
    # We need to find n in [1000, 1000000] that minimizes the number of distinct
    # values in {small_parts[n+1 - start_val], ..., small_parts[n+24 - start_val]}
    # i.e., small_parts[n], small_parts[n+1], ..., small_parts[n+23]
    # where n ranges from 1000 to 1000000, and small_parts is indexed from 0
    # corresponding to start_val = 1001
    
    # So for n, the 24 values are small_parts[n-1001 : n-1001+24]
    # Wait: n+1 corresponds to index (n+1) - 1001 = n - 1000
    # n+24 corresponds to index (n+24) - 1001 = n - 977
    
    # Let me re-verify:
    # start_val = 1001
    # small_parts[0] = small part of 1001
    # small_parts[i] = small part of (1001 + i)
    # For n, we want small parts of n+1, n+2, ..., n+24
    # n+1 = 1001 + (n+1-1001) = 1001 + (n-1000)
    # So index for n+1 is n-1000
    # Index for n+24 is n-1000+23 = n-977
    
    # n ranges from 1000 to 1000000
    # Index range: n-1000 ranges from 0 to 999000
    # n-977 ranges from 23 to 999023
    # total = 1000024, so max index is 1000023. n-977 max is 999023 < 1000024. OK.
    
    best_n = 1000
    best_score = 25
    
    start_time = time.time()
    deadline = start_time + 80
    
    # Use a sliding window approach with a frequency count
    # To speed up, we can use numpy for the window operations
    
    # Precompute a mapping from small_part values to compact IDs
    unique_parts = np.unique(small_parts)
    # Create a dictionary mapping
    part_to_id = {v: i for i, v in enumerate(unique_parts)}
    
    # Convert small_parts to IDs
    ids = np.array([part_to_id[v] for v in small_parts], dtype=np.int32)
    
    num_unique = len(unique_parts)
    
    # Sliding window distinct count
    # For each window of size 24, count distinct values
    
    # Use a frequency array
    freq = np.zeros(num_unique, dtype=np.int32)
    
    # Initialize the first window
    for i in range(24):
        freq[ids[i]] += 1
    
    distinct_count = np.count_nonzero(freq)
    
    if distinct_count < best_score:
        best_score = distinct_count
        best_n = 1000
    
    if best_score == 1:
        return best_n
    
    # Slide the window
    for n in range(1001, 1000001):
        if time.time() > deadline:
            break
        
        # Remove element at index n-1000-1 = n-1001
        # Add element at index n-977
        remove_idx = n - 1001
        add_idx = n - 977
        
        old_id = ids[remove_idx]
        freq[old_id] -= 1
        if freq[old_id] == 0:
            distinct_count -= 1
        
        new_id = ids[add_idx]
        if freq[new_id] == 0:
            distinct_count += 1
        freq[new_id] += 1
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:
                break
    
    return best_n
