"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    
    # Determine the maximum value we need to sieve
    max_val = max(hi for _, hi in P['gaps'])
    
    # Linear Sieve to compute Least Prime Factor (LPF) for all numbers up to max_val
    # lpf[i] will store the smallest prime factor of i
    lpf = [0] * (max_val + 1)
    primes = []
    lpf[1] = 1
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            lpf[i] = i
            primes.append(i)
        for p in primes:
            if p > lpf[i] or i * p > max_val:
                break
            lpf[i * p] = p

    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # We need an integer m such that lo < m < hi and lpf[m] >= gap
        # Note: if m is prime, lpf[m] = m. Since m < hi <= 10289, and gap is small (2 to 28),
        # if m is prime, lpf[m] = m. We need m >= gap.
        # Since m > lo >= 10007 and gap <= 28, any prime m will satisfy lpf[m] >= gap.
        # So any prime in the interval is a valid witness.
        # Also, composite numbers with no prime factor < gap are valid.
        
        found = False
        # Search from the middle outwards might be faster if witnesses are sparse?
        # Or just search linearly. The interval is small (< 30 numbers).
        # Linear search is fine.
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                found = True
                break
        
        # If not found, it remains None.
        # The task says "None never asserts nonexistence", so we just return None if we can't find one.
        # Given the speed, if it's not found, it likely doesn't exist or we missed it?
        # No, the check is exact.
        
    return result
# EVOLVE-BLOCK-END
