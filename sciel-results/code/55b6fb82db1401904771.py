"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin from 60s limit

    # Generate small primes up to 102 (sqrt(10289) approx 101.4)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    small_primes = sieve(102)

    def least_prime_factor(m, primes):
        if m == 1:
            return 1
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                return p
        return m

    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # Search for m in (lo, hi)
        # Optimization: We can skip even numbers if gap > 2, because LPF(even)=2.
        # If gap == 2, m is even, LPF=2. Condition 2>=2 holds.
        # If gap > 2, we need LPF >= gap > 2. So LPF cannot be 2.
        # Thus, for gap > 2, m must be odd.
        
        if gap == 2:
            m = lo + 1
            # m is even, LPF is 2. 2 >= 2 is true.
            result[i] = m
            continue
        
        # For gap > 2, only check odd m
        # m must be strictly between lo and hi
        # lo is prime (odd for lo>2). hi is prime (odd).
        # lo+1 is even. lo+2 is odd.
        # So we start at lo+2 if lo is odd.
        # All primes > 2 are odd.
        
        start_m = lo + 2
        if start_m % 2 == 0:
            start_m += 1 # Ensure odd, though lo+2 is odd if lo is odd.
            
        # Actually, if lo is odd, lo+1 is even, lo+2 is odd.
        # Range is lo+1 to hi-1.
        # Odds are lo+2, lo+4, ..., hi-2 (if hi-1 is even, hi is odd).
        
        for m in range(start_m, hi, 2):
            if time.time() > deadline:
                break
                
            lpf = least_prime_factor(m, small_primes)
            if lpf >= gap:
                result[i] = m
                break
                
    return result
# EVOLVE-BLOCK-END
