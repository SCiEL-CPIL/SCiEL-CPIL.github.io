import time
import sys
import numpy as np
from math import isqrt
from collections import defaultdict

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form a^2 * b^3 where b is square-free.
    """
    if limit < 1:
        return []
    
    powerful = set()
    powerful.add(1)
    
    # b must be square-free and b^3 <= limit
    # max_b such that b^3 <= limit
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    if max_b < 1:
        max_b = 1

    # We need to check if b is square-free.
    # Primes up to isqrt(max_b) are sufficient to check square-freeness of b.
    # If p^2 divides b, then p <= sqrt(b) <= sqrt(max_b).
    sieve_limit = isqrt(max_b) + 1
    if sieve_limit < 2:
        primes = []
    else:
        is_prime = bytearray(b'\x01') * (sieve_limit + 1)
        is_prime[0] = 0
        if sieve_limit >= 1:
            is_prime[1] = 0
        for i in range(2, isqrt(sieve_limit) + 1):
            if is_prime[i]:
                for j in range(i * i, sieve_limit + 1, i):
                    is_prime[j] = 0
        primes = [i for i in range(2, sieve_limit + 1) if is_prime[i]]
        
    for b in range(1, max_b + 1):
        # Check if b is square-free
        t = b
        is_sf = True
        for p in primes:
            pp = p * p
            if pp > t:
                break
            if t % pp == 0:
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
    # 150s limit. Reserve 10s for safety and extraction.
    deadline = start_time + 140
    
    # Strategy:
    # 1. Generate powerful numbers up to a limit L.
    # 2. Use a dictionary to count sums of pairs (a, b) where a <= b and a+b <= 2L.
    #    This avoids the memory explosion of a dense array of size 2L.
    # 3. Find the N with the maximum count.
    # 4. Extract the pairs.
    
    # The number of powerful numbers up to L is roughly L^(1/3) * log(L) or similar?
    # Actually, the count of powerful numbers <= x is ~ C * x^(1/3).
    # For L = 10^9, count is ~ 1000 * C. Let's say ~5000.
    # Pairs: ~ 2500 * 5000 / 2 = 6.25 * 10^6. This is manageable in a dict.
    # For L = 10^12, count is ~ 10^4. Pairs ~ 5 * 10^7. Might be slow in Python but maybe okay?
    # Let's stick to L where K is small enough that O(K^2) is fast.
    # If K ~ 5000, K^2/2 = 1.25 * 10^7. In Python, a loop of 10M iterations takes ~1s.
    # So L = 10^9 is safe.
    # Can we go higher?
    # L = 10^10 -> K ~ 10^4? K^2/2 = 5 * 10^7. ~5s.
    # L = 10^11 -> K ~ 2 * 10^4? K^2/2 = 2 * 10^8. ~20s.
    # Let's try L = 5 * 10^9 first.
    
    limits_to_try = [5_000_000_000, 2_000_000_000, 1_000_000_000, 500_000_000]
    
    best_result = {'N': 5, 'pairs': [[1, 4]]}
    
    for limit in limits_to_try:
        try:
            # Check time before starting
            if time.time() > deadline - 30:
                break
                
            # Generate powerful numbers
            gen_start = time.time()
            powerful = _generate_powerful_numbers(limit)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 20:
                # Not enough time left for counting
                continue
                
            K = len(powerful)
            if K == 0:
                continue
                
            pow_arr = np.array(powerful, dtype=np.int64)
            
            # Count sums using a dictionary
            # To optimize, we can use a list of counts if the range is small,
            # but the range is 2*limit which is too big for a list.
            # Dict is the way.
            
            # Optimization: Use a local variable for the dict
            counts = defaultdict(int)
            
            # Iterate over all pairs i <= j
            # To speed up, we can use numpy for the inner loop?
            # No, Python loop is fine if K is small enough.
            # Let's check K.
            
            # If K is too large, we might time out.
            # Estimated time ~ K^2 / 2 * 100ns = K^2 * 50ns.
            # If K = 10000, 10^8 * 50ns = 5s.
            # If K = 20000, 4*10^8 * 50ns = 20s.
            
            # Let's add a check on K
            if K > 15000:
                # Too slow for pure Python nested loop
                # Try a smaller limit? Or break?
                # The next limit is smaller, so it will be faster.
                continue
            
            # Nested loop
            # We can break early if time is running out?
            # Better to just do it and hope.
            
            for i in range(K):
                a = pow_arr[i]
                # a + b <= 2 * limit is always true since b <= limit
                # We just need to count all pairs.
                
                # Optimization: If a + a > some threshold? No, we want the max count.
                
                # Inner loop
                # Using list comprehension or loop?
                # Loop is clearer.
                
                # We can vectorize the inner loop?
                # sums = a + pow_arr[i:]
                # Then update counts?
                # Updating a dict with numpy array is not direct.
                # np.unique(sums, return_counts=True) gives unique sums and their counts in the slice.
                # This is O(M log M) for the slice of size M = K - i.
                # Total time sum_{i=0}^{K} (K-i) log(K-i) ~ K^2 log K / 2.
                # This might be slower than O(K^2) simple loop if log K is large?
                # But np.unique is implemented in C.
                # Let's try this approach. It might be much faster.
                
                # However, np.unique returns sorted unique values.
                # We need to add to the global counts.
                
                if i < K:
                    sums = a + pow_arr[i:]
                    unique_sums, unique_counts = np.unique(sums, return_counts=True)
                    
                    # Update counts dict
                    # This is a Python loop over unique_sums, which is smaller than K-i
                    # but could still be large.
                    # Actually, the number of unique sums is close to K-i because powerful numbers are sparse.
                    # So it's basically O(K^2) still.
                    
                    # Let's stick to the simple nested loop but optimized with local vars.
                    # Or maybe just the simple loop is fine for K < 15000.
                    
                    # Let's revert to simple loop for correctness and simplicity,
                    # and rely on the limit selection to keep K small.
                    
            # Actually, the np.unique approach adds overhead of creating the array and sorting.
            # Let's just use the simple loop.
            
            # Re-do the counting with simple loop
            counts = {}
            for i in range(K):
                a = pow_arr[i]
                for j in range(i, K):
                    b = pow_arr[j]
                    s = a + b
                    if s in counts:
                        counts[s] += 1
                    else:
                        counts[s] = 1
                        
            # Find best N
            if not counts:
                continue
                
            best_N = max(counts, key=counts.get)
            best_count = counts[best_N]
            
            # Check constraint N <= 10^18
            if best_N > 10**18:
                continue
            
            # Extract pairs
            pow_set = set(powerful)
            pairs = []
            
            half_N = best_N // 2
            
            # Iterate through powerful numbers up to half_N
            end_idx = np.searchsorted(pow_arr, half_N, side='right')
            
            for a in pow_arr[:end_idx]:
                b = best_N - a
                if b in pow_set:
                    pairs.append([int(a), int(b)])
            
            if pairs:
                if len(pairs) > len(best_result['pairs']):
                    best_result = {'N': best_N, 'pairs': pairs}
                    
            # Return immediately if we found a good result?
            # The score is the number of pairs.
            # Higher limit -> more powerful numbers -> potentially higher peak count.
            # But the peak count might not scale monotonically with limit.
            # However, usually larger sets have higher collision peaks.
            # Let's return the first valid result from the largest limit.
            return best_result

        except Exception as e:
            # Fallback to next
            continue
            
    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
