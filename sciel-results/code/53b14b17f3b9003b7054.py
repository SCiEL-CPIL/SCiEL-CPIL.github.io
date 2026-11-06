"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt
from collections import defaultdict

def generate_powerful(limit):
    """
    Generate all powerful numbers up to limit.
    Optimized using a sieve for square-free numbers.
    """
    pows = set([1])
    
    # Max b such that b^3 <= limit
    max_b = int(limit ** (1.0/3.0)) + 2
    
    # Sieve for square-free numbers up to max_b
    # is_square_free[n] is True if n is square-free
    is_sq_free = np.ones(max_b + 1, dtype=bool)
    is_sq_free[0] = False
    is_sq_free[1] = True
    
    # Sieve out multiples of p^2
    # We only need to check primes up to sqrt(max_b)
    # But a simple sieve marking multiples of i^2 for i=2..sqrt(max_b) works
    # Note: We mark i^2, 2*i^2, etc.
    # Actually, standard square-free sieve:
    # for i in range(2, isqrt(max_b) + 1):
    #     if is_sq_free[i]: # i is prime (or at least not marked by smaller square)
    #         # Mark multiples of i^2 as not square-free
    #         start = i*i
    #         is_sq_free[start::i*i] = False
    # Wait, this logic is slightly off for non-primes. 
    # If i is not square-free, i^2 is a multiple of some p^2, so it's already marked?
    # No, we need to iterate over all i, and if i is square-free, it doesn't matter.
    # We just want to mark numbers divisible by any square.
    # So iterate i from 2 to isqrt(max_b).
    # If i is prime, mark multiples of i^2.
    # We need a prime sieve first or just check if i is square-free?
    # Actually, if i is not square-free, i^2 is divisible by p^2 for some p<i.
    # So the multiples of i^2 are a subset of multiples of p^2.
    # So we only need to mark for square-free i? No, just for all i?
    # If we mark for all i, we might do redundant work but it's correct.
    # But we only need to mark for primes to be efficient?
    # Let's use a simple prime sieve up to isqrt(max_b) and mark multiples of p^2.
    
    limit_sq = isqrt(max_b)
    is_prime = np.ones(limit_sq + 1, dtype=bool)
    if limit_sq >= 0:
        is_prime[0] = False
    if limit_sq >= 1:
        is_prime[1] = False
    for p in range(2, limit_sq + 1):
        if is_prime[p]:
            # Mark multiples of p^2 in is_sq_free
            start = p * p
            if start <= max_b:
                is_sq_free[start::p] = False # Wait, step is p*p? No, multiples of p^2 are p^2, 2p^2, 3p^2...
                # is_sq_free[start::p*p] = False
                is_sq_free[start::p*p] = False
            # Mark p as prime (it already is), and mark multiples of p in is_prime as False
            # We don't strictly need is_prime for the powerful gen, just is_sq_free.
            # But we need to know which i are prime to do the marking?
            # No, the marking `is_sq_free[p*p::p*p]` works for any p.
            # But if p is composite, say p=4, p^2=16. 16 is already marked by p=2 (4,8,12,16...).
            # So we can just iterate p from 2 to limit_sq and mark.
            # To be safe and fast, let's just iterate all i from 2 to limit_sq.
            # But we already did the sieve logic inside the loop above?
            # Let's rewrite the sieve part clearly.
            
    # Re-doing the sieve properly
    is_sq_free = np.ones(max_b + 1, dtype=bool)
    is_sq_free[0] = False
    for i in range(2, isqrt(max_b) + 1):
        if is_sq_free[i]: # i is square-free, so it's a candidate for a prime factor?
             # Actually, if i is square-free, it might be composite (e.g., 6).
             # But 6^2 = 36 is divisible by 2^2 and 3^2.
             # So marking multiples of i^2 is redundant if i is composite.
             # We only need to mark for primes.
             # So we need a prime check.
             pass

    # Let's use a simpler, robust method for small max_b (up to ~10^5)
    # Just check each b for square-free property using precomputed primes.
    
    # Generate primes up to isqrt(max_b)
    limit_prime = isqrt(max_b)
    primes = []
    if limit_prime >= 2:
        sieve = np.ones(limit_prime + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(limit_prime) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    pows = set([1])
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_sf = True
        for p in primes:
            if p * p > b:
                break
            if b % (p * p) == 0:
                is_sf = False
                break
        if not is_sf:
            continue
            
        b_cubed = b ** 3
        if b_cubed > limit:
            break
            
        max_a = isqrt(limit // b_cubed)
        # Generate a^2 * b^3 for a=1..max_a
        # We can vectorize this?
        # a_vals = np.arange(1, max_a + 1)
        # n_vals = (a_vals ** 2) * b_cubed
        # pows.update(n_vals.tolist())
        
        # For small max_a, loop is fine. For large, vectorize.
        if max_a > 1000:
            a_vals = np.arange(1, max_a + 1, dtype=np.int64)
            n_vals = (a_vals * a_vals) * b_cubed
            # Filter <= limit (should be all)
            pows.update(n_vals.tolist())
        else:
            for a in range(1, max_a + 1):
                n = a * a * b_cubed
                if n <= limit:
                    pows.add(n)
                else:
                    break
    return sorted(pows)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try a larger limit. 
    # With optimized generation, we can try 10^9 or 10^10.
    # Let's estimate time for generation of 10^9.
    # max_b = 1000.
    # Number of square-free b ~ 6/pi^2 * 1000 ~ 600.
    # For each b, max_a ~ sqrt(10^9 / b^3).
    # Sum of max_a over b.
    # This is roughly integral sqrt(L/b^3) db ~ L^0.5 * b^-1.5 ~ L^0.5.
    # So total numbers ~ K ~ 0.8 * sqrt(10^9) ~ 25000.
    # Generation should be fast.
    
    limit = 10**9
    powerful_list = generate_powerful(limit)
    powerful_set = set(powerful_list)
    K = len(powerful_list)
    
    # If K is too small, try larger limit? 
    # If K is too large, we might timeout in counting.
    # Let's check K.
    # For 10^9, K ~ 25000.
    # K^2/2 = 3.1e8 sums.
    # Pure Python loop: ~30-60s.
    # NumPy chunked: ~5-10s.
    
    # Use NumPy for counting
    P = np.array(powerful_list, dtype=np.int64)
    
    # We will count sums in chunks to avoid memory explosion
    # Chunk size for i: M
    # Memory: M * K * 8 bytes
    # If K=25000, M=1000 -> 200 MB. OK.
    M = 1000
    
    # We need counts for all possible sums.
    # Max sum is 2*limit.
    # We can use a dictionary or a large array.
    # Array of size 2*limit+1 is too big (2*10^9).
    # So we must use a dictionary or sparse method.
    # But np.bincount requires an array of indices.
    # If we collect all sums in a list/array, it's K^2/2 elements.
    # 3e8 elements. 
    # Creating an array of 3e8 int64 is 2.4 GB. Too much.
    
    # So we must count in a way that doesn't store all sums at once.
    # We can use a dictionary for counts, but updating it 3e8 times in Python is slow.
    # We can use np.unique on chunks?
    # For each chunk of M rows, we have M*K sums.
    # We can flatten, then use np.unique(return_counts=True) on that chunk.
    # Then add to a global dictionary.
    
    global_counts = defaultdict(int)
    
    for start_i in range(0, K, M):
        end_i = min(start_i + M, K)
        # Get the chunk of P for i
        P_i = P[start_i:end_i]
        
        # For each i in this chunk, we need j >= i.
        # This is tricky with vectorization because the lower bound of j changes.
        # We can compute all sums P_i[:, None] + P[None, :]
        # Then mask out j < i.
        # But creating the full M x K matrix is fine.
        
        # Create sums matrix
        # P_i shape (M, 1), P shape (1, K)
        sums = P_i[:, None] + P[None, :]
        
        # We need to zero out or mask where j < i.
        # For row r in 0..M-1, i = start_i + r.
        # We need columns j >= i.
        # So for row r, columns 0 to i-1 are invalid.
        
        # We can iterate over rows in the chunk to handle the mask?
        # No, that defeats the purpose.
        
        # Alternative: Just compute all sums P_i + P, and then subtract the counts for j < i?
        # Or simply:
        # The condition is i <= j.
        # If we compute all sums for i in [start_i, end_i) and all j in [0, K),
        # we are counting (i, j) and (j, i) if j < i.
        # Since we want distinct pairs a<=b, and we are iterating i as the smaller index?
        # No, P is sorted. So P[i] <= P[j] iff i <= j.
        # So we only want j >= i.
        
        # Let's process row by row within the chunk to apply the mask correctly,
        # but vectorize the counting for each row?
        # For a single i, sums are P[i] + P[i:].
        # This is a vector of size K-i.
        # We can use np.bincount on this vector?
        # But doing this M times in a Python loop inside the chunk loop?
        # That is M iterations of vectorized ops.
        # M=1000. K=25000.
        # 1000 * (25000 ops) = 25e6 ops. Fast.
        
        for r in range(end_i - start_i):
            i_idx = start_i + r
            # Sums with j >= i_idx
            # P[i_idx:] are the valid b's
            valid_b = P[i_idx:]
            sums_row = P[i_idx] + valid_b
            
            # Count these sums
            # np.unique(sums_row, return_counts=True)
            # This creates a new array each time.
            # Is there a faster way?
            # np.bincount(sums_row) requires max value.
            # Max sum is ~2*10^9. Array too big.
            
            # So np.unique is the way.
            unique_sums, counts = np.unique(sums_row, return_counts=True)
            
            # Add to global counts
            # Loop over unique_sums (small number compared to K)
            for s, c in zip(unique_sums, counts):
                global_counts[s] += c
        
        # Time check
        if time.time() > deadline - 20:
            break
            
    if not global_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Find best N
    best_N = max(global_counts.keys(), key=lambda n: global_counts[n])
    best_count = global_counts[best_N]
    
    # Collect pairs
    pairs = []
    for a in powerful_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}

# EVOLVE-BLOCK-END
