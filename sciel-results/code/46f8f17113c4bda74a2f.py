import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    
    def phi(n):
        if n == 1: return 1
        res = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                res -= res // p
                while temp % p == 0:
                    temp //= p
            if temp == 1:
                break
        if temp > 1:
            res -= res // temp
        return res

    best_k = 1
    best_s = 1
    
    # Helper to find best run in a range
    def find_best(L, R):
        nonlocal best_k, best_s
        # We compute phi for L..R
        # But R-L can be 10^6.
        # We want to find max k such that phi(L+i)..phi(L+i+k-1) are distinct.
        # We can do this by sliding a window.
        # However, computing phi for 10^6 numbers takes time.
        # Let's just compute them into a list.
        
        # Optimization: If we are running out of time, we might not finish the range.
        # But 10^6 is a fixed chunk.
        
        phis = []
        for n in range(L, R + 1):
            phis.append(phi(n))
            
        # Now find longest distinct subarray
        # Using a sliding window with a set
        seen = set()
        left = 0
        max_len = 0
        max_start = L
        for right in range(len(phis)):
            val = phis[right]
            while val in seen:
                seen.remove(phis[left])
                left += 1
            seen.add(val)
            if right - left + 1 > max_len:
                max_len = right - left + 1
                max_start = L + left
                
        if max_len > best_k:
            best_k = max_len
            best_s = max_start

    # Search Strategy
    # 1. Check a small range [1, 10000] just in case
    if time.time() < deadline - 10:
        find_best(1, 10000)
        
    # 2. Random large segments
    segment_size = 10**6
    search_lo = 10**11
    search_hi = 10**12 - segment_size
    
    while time.time() < deadline - 5.0:
        L = random.randint(search_lo, search_hi)
        find_best(L, L + segment_size - 1)
        
        # If we found a very good one, maybe we should scan neighbors?
        # The find_best already scans the whole segment.
        
    # 3. Polish
    if time.time() < deadline - 1.0 and best_k > 1:
        # Check around best_s
        # We already have the phis from the last segment? No, they are local.
        # Just re-check a small window around best_s
        # Actually, find_best on a small range around best_s
        local_lo = max(1, best_s - 100)
        local_hi = best_s + 100 + best_k
        find_best(local_lo, local_hi)

    return {'start': best_s, 'length': best_k}
