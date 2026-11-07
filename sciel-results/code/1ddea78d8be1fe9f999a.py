"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

# We need primes up to the maximum gap to test divisibility
MAX_GAP = max(hi - lo for lo, hi in P['gaps'])
# The largest gap is 28 (10273 to 10289).
# Primes < 28 are: 2, 3, 5, 7, 11, 13, 17, 19, 23
PRIMES = sieve(100)

# Pre-compute LPF for the entire range of interest [10007, 10289]
# Actually, we just need to check candidates.
# The strategy from the directive: "Use a segmented sieve to compute the least prime factor (LPF) for all integers in the range [10008, 10288] in a single pass"
# Let's implement this.

MIN_VAL = min(lo for lo, _ in P['gaps'])
MAX_VAL = max(hi for _, hi in P['gaps'])

# Range is small (~280 numbers), so a simple sieve for LPF over this range is very fast.
# However, LPF of a number n is the smallest prime p such that n % p == 0.
# If no such p <= sqrt(n) exists, LPF is n itself (if n is prime).
# But here n is around 10000, sqrt(n) is ~100.
# The condition is LPF(m) >= gap.
# This is equivalent to: m is not divisible by any prime p < gap.

# Let's precompute a "validity" mask or just the LPF for each number in the range.
# Since the range is small, we can just compute LPF for every number in [MIN_VAL, MAX_VAL].

def compute_lpf_range(start, end):
    """
    Computes the least prime factor for all integers in [start, end].
    Returns a dictionary {n: lpf(n)} or a list offset by start.
    """
    size = end - start + 1
    lpf = [0] * size
    
    # We need primes up to sqrt(end)
    limit = int(end**0.5) + 1
    primes = sieve(limit)
    
    # Initialize LPF for all numbers
    for i in range(size):
        lpf[i] = start + i # Default to itself (prime case)
        
    # Sieve-like pass to mark smallest prime factors
    for p in primes:
        if p * p > end:
            break
        # Find the first multiple of p in [start, end]
        # first = start + ((p - start % p) % p)
        first = start
        if first % p != 0:
            first += p - (first % p)
            
        for m in range(first, end + 1, p):
            idx = m - start
            if lpf[idx] == 0 or lpf[idx] > p: # Actually, since we iterate primes in ascending order, the first one we hit is the smallest.
                # But wait, if lpf[idx] is already set, it must be a smaller prime?
                # No, we iterate p in ascending order. So the first p that divides m is the LPF.
                # So we should only set it if it hasn't been set yet?
                # Or simply: if lpf[idx] is still 0 (unvisited by smaller primes), set it.
                # But 0 is a valid LPF? No, LPF >= 2.
                # Let's use 0 as "unset".
                if lpf[idx] == 0:
                    lpf[idx] = p
    
    # For numbers that remain 0, they are prime (or 1, but our range is > 10000)
    # Wait, the logic above:
    # We iterate primes p.
    # For each multiple m of p, if lpf[m-start] is 0, set it to p.
    # This works because primes are processed in increasing order.
    # Any composite number will be hit by its smallest prime factor first.
    # Prime numbers will never be hit by any p <= sqrt(n), so they remain 0.
    # We need to distinguish "0" (prime) from "unset".
    # Actually, if a number is prime, its LPF is itself.
    # So after the loop, if lpf[i] == 0, set it to start + i.
    
    for i in range(size):
        if lpf[i] == 0:
            lpf[i] = start + i
            
    return lpf

# Precompute LPF for the entire range
LPF_ARRAY = compute_lpf_range(MIN_VAL, MAX_VAL)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin out of 90s
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        # Search for witness m in (lo, hi) such that LPF[m] >= gap
        # Since LPF_ARRAY is indexed by value - MIN_VAL
        found = False
        for m in range(lo + 1, hi):
            lpf_m = LPF_ARRAY[m - MIN_VAL]
            if lpf_m >= gap:
                result[i] = m
                found = True
                break
        
        if not found:
            result[i] = None
            
    # Ensure all entries are proper Python ints or None
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            final_result.append(int(entry))
    
    return final_result

# EVOLVE-BLOCK-END
