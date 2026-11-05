import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe
    
    # We want to minimize max(lpf(n), lpf(n+1))
    # This is equivalent to finding n such that both n and n+1 are y-smooth
    # for the smallest possible y.
    
    # Strategy:
    # 1. Precompute smallest prime factor (SPF) or largest prime factor (LPF) for all numbers up to 1,000,000.
    #    Actually, calculating LPF directly via sieve is efficient.
    # 2. Iterate through possible y (largest prime factor) starting from small primes.
    #    But iterating y from 2 upwards and checking if any pair (n, n+1) is y-smooth is O(N * pi(N)) which might be too slow if we do it naively.
    #    
    # Better Strategy based on the directive:
    # "Construct a bitmask or boolean array marking all y-smooth numbers up to 10^6 by multiplying combinations of small primes, then scan for the longest consecutive run..."
    # However, the objective is to minimize the *maximum* of the two LPFs.
    # Let L(n) = max(lpf(n), lpf(n+1)). We want to minimize L(n).
    #
    # Let's compute lpf[i] for all i in [1, 1000000].
    # Then compute L[n] = max(lpf[n], lpf[n+1]) for n in [10000, 999999].
    # Then find the min L[n] and the corresponding n.
    #
    # Complexity of computing LPF:
    # A modified sieve can compute the largest prime factor in O(N log log N).
    
    N = 1000000
    lpf = [0] * (N + 1)
    
    # Sieve for largest prime factor
    for i in range(2, N + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i # Since we iterate i in increasing order, the last divisor that hits j is the largest prime factor? 
                           # Wait. If i is prime, it divides j. We update lpf[j] = i.
                           # Since i increases, the final value of lpf[j] will be the largest prime factor of j.
                           # This works.
    
    # Now find the best n
    best_n = 10000
    best_val = float('inf')
    
    # We can optimize the search by checking candidates.
    # But a simple linear scan is O(N) which is 10^6 operations, very fast.
    for n in range(10000, 1000000):
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
            # If best_val is 2, it's the minimum possible (since primes are >= 2). 
            # But n and n+1 can't both be powers of 2.
            # If best_val is small, we might break early? 
            # The minimum possible value for max(lpf(n), lpf(n+1)) is at least 2.
            # Actually, if n is a power of 2 and n+1 is a power of 3? No, 2^k + 1 = 3^m has few solutions.
            # Let's just do the full scan, it's fast enough.
    
    # Check time just in case
    if time.time() > deadline:
        pass # We are already done with the main logic which is fast.
    
    return best_n
