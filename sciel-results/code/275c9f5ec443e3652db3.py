import heapq
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Primes up to 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def is_smooth(m, primes):
        for p in primes:
            if p * p > m:
                break
            while m % p == 0:
                m //= p
        return m == 1

    def largest_prime_factor(m, primes):
        lpf = 1
        temp = m
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                lpf = p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            lpf = temp
        return lpf

    # Generate y-smooth numbers up to 10^6 using a heap
    # We want to find n in [10000, 1000000] such that n and n+1 are y-smooth
    # We'll generate all 100-smooth numbers up to 10^6
    
    LIMIT = 1000000
    smooth_nums = set()
    heap = [1]
    smooth_nums.add(1)
    
    while heap:
        curr = heapq.heappop(heap)
        if curr > LIMIT:
            break
        for p in primes:
            nxt = curr * p
            if nxt > LIMIT:
                break
            if nxt not in smooth_nums:
                smooth_nums.add(nxt)
                heapq.heappush(heap, nxt)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Find the best n
    best_n = 10000
    best_lpf = float('inf')
    
    # We iterate through potential n. Since n and n+1 must be in smooth_nums,
    # we can just iterate through smooth_nums and check if n+1 is there.
    for n in smooth_nums:
        if n < 10000:
            continue
        if n + 1 not in smooth_nums:
            continue
            
        # Calculate largest prime factor of n(n+1)
        lpf_n = largest_prime_factor(n, primes)
        lpf_n1 = largest_prime_factor(n + 1, primes)
        current_lpf = max(lpf_n, lpf_n1)
        
        if current_lpf < best_lpf:
            best_lpf = current_lpf
            best_n = n
            
        if time.time() > deadline:
            break
            
    return best_n
# EVOLVE-BLOCK-END
