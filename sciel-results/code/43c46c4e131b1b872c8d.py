"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import reduce
from math import gcd

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _build_wheel(modulus, forbidden_primes):
    """
    Build sorted list of residues r in [0, modulus) such that
    gcd(r, modulus) == 1 (i.e., r is not divisible by any prime < max_gap).
    Actually, we need: r not divisible by any p in forbidden_primes.
    """
    if not forbidden_primes:
        # No forbidden primes: every residue works, but we just need to return
        # a list that covers all residues. For efficiency, return [0] and handle
        # specially, or return range. Let's return a marker.
        return None
    
    valid_residues = []
    for r in range(modulus):
        valid = True
        for p in forbidden_primes:
            if r % p == 0:
                valid = False
                break
        if valid:
            valid_residues.append(r)
    return valid_residues

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n
    
    # Determine the maximum gap size
    max_gap = max(hi - lo for lo, hi in gaps)
    
    # Precompute all primes up to max_gap - 1
    all_primes = _sieve_primes(max_gap - 1)
    
    # For each distinct gap size, precompute:
    # 1. The list of forbidden primes (primes < gap_size)
    # 2. The modulus = product of forbidden primes (or lcm, but since they're distinct primes, product = lcm)
    # 3. The valid residues modulo that modulus
    gap_data = {}
    for lo, hi in gaps:
        g = hi - lo
        if g not in gap_data:
            forbidden = [p for p in all_primes if p < g]
            if not forbidden:
                gap_data[g] = {'forbidden': forbidden, 'modulus': 1, 'residues': None}
            else:
                modulus = 1
                for p in forbidden:
                    modulus *= p
                residues = _build_wheel(modulus, forbidden)
                gap_data[g] = {'forbidden': forbidden, 'modulus': modulus, 'residues': residues}
    
    # Process each gap
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        
        g = hi - lo
        data = gap_data[g]
        forbidden = data['forbidden']
        modulus = data['modulus']
        residues = data['residues']
        
        if not forbidden:
            # No forbidden primes: any integer in (lo, hi) works
            result[i] = int(lo + 1)
            continue
        
        # Find the smallest m in (lo, hi) such that m mod modulus is in residues
        # Start from lo+1
        start = lo + 1
        end = hi  # exclusive
        
        # Find the first valid candidate >= start
        # Compute start mod modulus
        r_start = start % modulus
        
        # Find the smallest residue >= r_start in the sorted residues list
        # If none, wrap around to the smallest residue + modulus
        import bisect
        idx = bisect.bisect_left(residues, r_start)
        
        if idx < len(residues):
            # There's a residue >= r_start
            delta = residues[idx] - r_start
            candidate = start + delta
        else:
            # Wrap around: use the smallest residue in the next cycle
            delta = residues[0] + modulus - r_start
            candidate = start + delta
        
        # Now candidate is the first valid number >= start
        # Check if candidate < hi
        if candidate < hi:
            result[i] = int(candidate)
        else:
            # No valid witness found in this gap
            # Try a more thorough search as fallback (shouldn't happen often)
            for m in range(start, hi):
                valid = True
                for p in forbidden:
                    if m % p == 0:
                        valid = False
                        break
                if valid:
                    result[i] = int(m)
                    break
    
    # Ensure types are correct: Python int or None
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result
# EVOLVE-BLOCK-END
