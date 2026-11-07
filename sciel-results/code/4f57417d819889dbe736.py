"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from concurrent.futures import ThreadPoolExecutor
import numpy as np

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin before 90s kill

    # Precompute smallest prime factor for numbers up to max candidate + 1
    max_candidate = max(hi for lo, hi in P['gaps'])
    spf = list(range(max_candidate + 1))
    for i in range(2, isqrt(max_candidate) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_candidate + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute list of primes up to max gap size
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = [i for i in range(2, max_gap) if spf[i] == i]

    def find_witness(lo, hi):
        """Find a witness m in (lo, hi) with lpf(m) >= hi-lo."""
        gap = hi - lo
        # Primes that would disqualify a candidate: primes < gap
        disqualify_primes = [p for p in small_primes if p < gap]
        
        # Try all candidates in the gap
        # Use vectorized approach: check which candidates are not divisible by any disqualifying prime
        candidates = list(range(lo + 1, hi))
        if not candidates:
            return None
        
        # For each candidate, check if it's divisible by any prime < gap
        # If not, its LPF >= gap (or it's prime)
        # We can do this efficiently by marking candidates divisible by each small prime
        
        n = len(candidates)
        valid = np.ones(n, dtype=bool)
        
        for p in disqualify_primes:
            # Mark candidates divisible by p as invalid
            # candidates[i] % p == 0
            # Use vectorized modulo
            valid &= (np.array(candidates) % p != 0)
            if not np.any(valid):
                return None
        
        if np.any(valid):
            # Return the first valid candidate
            idx = np.argmax(valid)
            return int(candidates[idx])
        
        return None

    def find_witness_sequential(lo, hi):
        """Sequential fallback with middle-out heuristic."""
        gap = hi - lo
        mid = (lo + hi) // 2
        
        # Generate candidates in order of distance from mid
        for offset in range(0, (hi - lo) // 2 + 1):
            for m in [mid + offset, mid - offset]:
                if lo < m < hi:
                    lpf = spf[m] if m <= max_candidate else m
                    if lpf >= gap:
                        return int(m)
        
        # Exhaustive scan
        for m in range(lo + 1, hi):
            lpf = spf[m] if m <= max_candidate else m
            if lpf >= gap:
                return int(m)
        
        return None

    result = [None] * len(P['gaps'])

    # Use ThreadPoolExecutor for parallel gap searches
    def process_gap(i):
        lo, hi = P['gaps'][i]
        if time.time() > deadline:
            return i, None
        w = find_witness(lo, hi)
        if w is None:
            w = find_witness_sequential(lo, hi)
        return i, w

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_gap, i) for i in range(len(P['gaps']))]
        for f in futures:
            i, w = f.result()
            if w is not None:
                result[i] = w

    return result
# EVOLVE-BLOCK-END
