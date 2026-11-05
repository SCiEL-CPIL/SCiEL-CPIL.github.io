import time
import sys
import numpy as np
from math import isqrt

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    for p in small_primes:
        pp = p * p
        if pp > temp:
            break
        if temp % pp == 0:
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            return False
    if temp > 1:
        r = isqrt(temp)
        if r * r != temp:
            return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit using n = a^2 * b^3, b square-free."""
    powerful = set()
    powerful.add(1)
    max_b = int(limit ** (1.0 / 3.0)) + 2
    
    # Sieve of Eratosthenes up to max_b to get primes
    max_b_plus = max_b + 1
    if max_b_plus > 1:
        is_prime = bytearray(b'\x01') * (max_b_plus)
        is_prime[0] = 0
        if max_b_plus > 1:
            is_prime[1] = 0
        for i in range(2, isqrt(max_b) + 1):
            if is_prime[i]:
                for j in range(i * i, max_b_plus, i):
                    is_prime[j] = 0
        small_primes = [i for i in range(2, max_b_plus) if is_prime[i]]
    else:
        small_primes = []
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
        if not is_sf:
            continue
        
        b3 = b * b * b
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # Strategy: Generate powerful numbers, then use batched numpy operations
    # to count sums efficiently. The key insight from inspirations is that
    # batched np.unique on concatenated sums reduces Python loop overhead.
    
    # Let's try a larger limit. The number of powerful numbers up to L is ~2.16*sqrt(L).
    # L = 5*10^9 -> K ~ 480,000 -> K^2 ~ 2.3*10^11 (too many for memory)
    # L = 10^9  -> K ~ 216,000 -> K^2 ~ 4.7*10^10
    # L = 5*10^8 -> K ~ 153,000 -> K^2 ~ 2.3*10^10
    
    # For batched approach, we process chunks of the powerful numbers list.
    # Each chunk: for each element in chunk, compute sum with all other elements.
    # Use np.add.outer for vectorized computation.
    
    pow_limit = 500_000_000  # 5 * 10^8
    
    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)
    
    if time.time() > deadline - 60:
        # Fallback to smaller limit
        pow_limit = 200_000_000
        powerful = _generate_powerful_numbers(pow_limit)
        K = len(powerful)
    
    pow_arr = np.array(powerful, dtype=np.int64)
    
    # Use a dictionary-like approach with numpy bincount
    # But bincount requires non-negative integers and the range must be manageable.
    # Max sum is 2 * pow_limit. For 5*10^8, that's 10^9 which is too large for bincount.
    
    # Alternative: use a hash map approach with numpy
    # Or: process in chunks and use np.unique to count
    
    # Better approach: use the fact that we want to find N with many representations.
    # We can sample random pairs and count frequencies, but that's probabilistic.
    
    # Actually, let's use a different strategy: 
    # 1. Generate powerful numbers
    # 2. For each powerful number a, compute N - a for candidate N values
    # 3. Count how many a give powerful N - a
    
    # But we don't know N in advance. So we need to find N that maximizes the count.
    
    # Efficient approach: 
    # - Use numpy to compute all pairwise sums in batches
    # - Use np.unique with return_counts to count frequencies
    # - Find the sum with highest count
    
    # Memory consideration: K^2 elements. For K=153,000, that's ~2.3*10^10, too large.
    # So we need to process in chunks.
    
    # Chunk-based approach:
    # - Split pow_arr into chunks of size C
    # - For each chunk, compute sums with all elements (or with other chunks)
    # - Accumulate counts using a dictionary or sparse representation
    
    # Since max_sum can be up to 10^9, we can't use a dense array.
    # Use a dictionary to accumulate counts.
    
    from collections import defaultdict
    
    sum_counts = defaultdict(int)
    
    # Process in chunks to manage memory
    # For each pair (i, j) with i <= j, compute sum and count it.
    # To avoid double counting, only consider i <= j.
    
    # Chunk size: balance between memory and number of chunks
    # Each chunk: C * K sums. For C=1000, K=153000: 153M sums per chunk.
    # np.unique on 153M elements takes ~1-2 seconds.
    # Number of chunks: K/C = 153. Total time: ~153 * 2 = 306 seconds. Too slow.
    
    # Need a better approach. Let's think differently.
    
    # Observation: The number of powerful numbers up to L is ~2.16*sqrt(L).
    # For L = 10^18, that's ~2.16 * 10^9, which is too many to generate.
    # So we're limited to L ~ 10^9 or so.
    
    # Alternative: use the structure of powerful numbers.
    # Powerful numbers are of the form a^2 * b^3 where b is square-free.
    # This structure might allow more efficient counting.
    
    # For now, let's use a moderate limit and hope for the best.
    
    # Actually, let's try a smarter approach:
    # 1. Generate powerful numbers up to a moderate limit (say 10^8)
    # 2. Use numpy to compute all pairwise sums in batches
    # 3. Use np.unique to find the most common sum
    # 4. Verify the pairs
    
    pow_limit = 100_000_000  # 10^8
    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)
    pow_arr = np.array(powerful, dtype=np.int64)
    
    # Process in chunks
    # For each chunk of size C, compute sums with all K elements
    # Use np.add.outer(chunk, pow_arr) to get C*K sums
    # Then use np.unique to count
    
    # But we only want i <= j to avoid double counting.
    # Actually, for the purpose of finding the best N, double counting is fine
    # as long as we're consistent. We just want the N with the most representations.
    
    # Let's process in chunks of size 1000
    chunk_size = 1000
    sum_counts = {}
    
    for i in range(0, K, chunk_size):
        chunk = pow_arr[i:i+chunk_size]
        # Compute all sums: chunk[j] + pow_arr[k] for all j, k
        # This is C * K sums
        sums = np.add.outer(chunk, pow_arr).ravel()
        
        # Count unique values
        unique_sums, counts = np.unique(sums, return_counts=True)
        
        # Accumulate
        for s, c in zip(unique_sums, counts):
            sum_counts[int(s)] = sum_counts.get(int(s), 0) + int(c)
        
        if time.time() > deadline - 20:
            break
    
    if not sum_counts:
        return {'N': 5, 'pairs': [[1, 4]]}
    
    # Find the best N
    best_N = max(sum_counts, key=sum_counts.get)
    best_count = sum_counts[best_N]
    
    # Extract pairs
    pow_set = set(powerful)
    pairs = []
    
    half_N = best_N // 2
    for a in powerful:
        if a > half_N:
            break
        b = best_N - a
        if b in pow_set:
            if _is_powerful(a) and _is_powerful(b):
                pairs.append([a, b])
    
    if not pairs:
        return {'N': 5, 'pairs': [[1, 4]]}
    
    return {'N': best_N, 'pairs': pairs}
