"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    max_gap = max(hi - lo for lo, hi in P['gaps'])

    # Sieve for primes up to max_gap
    if max_gap >= 2:
        sieve = [True] * (max_gap + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_gap**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, max_gap + 1, i):
                    sieve[j] = False
        primes_up_to_max_gap = [i for i in range(2, max_gap + 1) if sieve[i]]
    else:
        primes_up_to_max_gap = []

    # Precompute forbidden primes for each gap size
    forbidden_primes = {}
    for g in range(2, max_gap + 1):
        fp = [p for p in primes_up_to_max_gap if p < g]
        forbidden_primes[g] = fp

    # Modular arithmetic coverage: for each gap size, precompute which residues
    # modulo the product of forbidden primes are "bad" (divisible by some forbidden prime)
    # We use a bitset approach: for each gap, mark bad residues mod LCM or use inclusion-exclusion
    # Actually, simpler: for each gap, precompute a set of "bad" residue classes mod M where M = product of forbidden primes
    # But M can be huge. Instead, for each candidate m, check m % p == 0 for p in fp.
    # Optimize: precompute for each gap a list of forbidden primes, and use a fast check.
    
    # Better optimization: for each gap size, precompute a "bad mask" using the Chinese Remainder Theorem
    # For a given gap g with forbidden primes fp = [p1, p2, ..., pk], a number m is bad if m ≡ 0 (mod pi) for some i.
    # The "good" residues modulo M = p1*p2*...*pk are those not divisible by any pi.
    # But M can be very large. For small gaps, fp is small.
    
    # Alternative: precompute for each gap a lookup table of valid residues modulo L = lcm(fp)
    # For small gaps, L is manageable.
    
    # Let's use a different approach: for each gap, precompute the set of "bad" residues modulo each prime,
    # then for a candidate m, check if m % p == 0 for any p in fp. This is what we already do.
    
    # Key insight: for each gap, the number of forbidden primes is small (primes < gap).
    # For gap=20, forbidden primes are [2,3,5,7,11,13,17,19] = 8 primes.
    # For gap=18, forbidden primes are [2,3,5,7,11,13,17] = 7 primes.
    # The check is at most 8-10 modular divisions per candidate, which is fast.
    
    # The real bottleneck is scanning candidates. Let's precompute for each gap
    # a periodic pattern of good/bad residues modulo M = product of forbidden primes.
    # But M can be huge (product of first 8 primes = 9699690).
    
    # Actually, for our purposes, the gaps are small (max ~20), so the number of
    # forbidden primes is small. The per-candidate check is fast enough.
    # Let's focus on a different optimization: precompute the "next good number" 
    # after any starting point for each gap pattern.
    
    # For each gap size g, the pattern of good/bad numbers repeats with period M_g = product of forbidden primes.
    # But M_g can be large. Instead, let's just precompute for each gap a sorted list of
    # "bad" offsets within one period, then for a given range [lo+1, hi-1], find the first good number.
    
    # Simpler optimization: for each gap, precompute a boolean array of length M_g indicating
    # which residues are good. Then for a range, we can quickly find the first good residue.
    
    # Let's try: for each distinct gap size, precompute the good residues modulo M_g.
    gap_patterns = {}
    for g in range(2, max_gap + 1):
        fp = forbidden_primes[g]
        if not fp:
            gap_patterns[g] = None  # all numbers are good
            continue
        
        # Compute M = product of forbidden primes
        M = 1
        for p in fp:
            M *= p
        
        # If M is too large, fall back to per-candidate checking
        if M > 10**7:
            gap_patterns[g] = ('large', fp)
            continue
        
        # Precompute good residues
        good = [True] * M
        for p in fp:
            for r in range(0, M, p):
                good[r] = False
        
        # Find all good residues
        good_residues = [r for r in range(M) if good[r]]
        gap_patterns[g] = ('small', M, good_residues)

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        pattern = gap_patterns[gap]
        
        if pattern is None:
            # All numbers are good
            result[i] = lo + 1
            continue
        
        if pattern[0] == 'large':
            # Fall back to per-candidate checking
            fp = pattern[1]
            for m in range(lo + 1, hi):
                is_valid = True
                for p in fp:
                    if m % p == 0:
                        is_valid = False
                        break
                if is_valid:
                    result[i] = m
                    break
        else:
            # Use precomputed pattern
            M = pattern[1]
            good_residues = pattern[2]
            
            # Find first good number in [lo+1, hi-1]
            start = lo + 1
            end = hi - 1
            
            # Compute start residue
            start_res = start % M
            
            # Check residues from start_res to end_res (wrapping around if needed)
            # First, check from start_res to M-1
            found = False
            for r in good_residues:
                if r >= start_res:
                    # Candidate: start + (r - start_res)
                    m = start + (r - start_res)
                    if m <= end:
                        result[i] = m
                        found = True
                        break
            
            if not found:
                # Wrap around: check residues from 0 to end_res
                end_res = end % M
                for r in good_residues:
                    if r <= end_res:
                        # The number would be start + (M - start_res) + r
                        m = start + (M - start_res) + r
                        if m <= end:
                            result[i] = m
                            found = True
                            break
            
            if not found:
                # Try a few more periods (should not happen for our data)
                for period in range(1, 3):
                    base = start + period * M
                    if base > end:
                        break
                    for r in good_residues:
                        m = base + r
                        if m <= end:
                            result[i] = m
                            found = True
                            break
                    if found:
                        break

    # Ensure all entries are Python int or None
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))

    return final_result
# EVOLVE-BLOCK-END
