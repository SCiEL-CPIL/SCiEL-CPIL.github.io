"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import bisect
from collections import defaultdict
from fractions import Fraction

# EVOLVE-BLOCK-START

MAX_N = 10_000_000
TARGET = Fraction(3, 2)

def _sieve_spf(limit):
    """Return smallest prime factor for each number up to limit."""
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _sigma_sieve(N, spf):
    """
    Compute sigma(k) for k=1..N using SPF-based multiplicative recurrence.
    sigma(n) = sigma(p^k) * sigma(m) where n = p^k * m and p ∤ m.
    Uses arrays p_power[n] and sigma_p[n] for the highest power of spf[n] dividing n.
    """
    sigma = [0] * (N + 1)
    sigma[1] = 1
    p_power = [0] * (N + 1)
    sigma_p = [0] * (N + 1)
    
    for n in range(2, N + 1):
        p = spf[n]
        if n % (p * p) == 0:
            prev = n // p
            p_power[n] = p_power[prev] * p
            sigma_p[n] = sigma_p[prev] + p_power[n]
            m = n // p_power[n]
            sigma[n] = sigma_p[n] * sigma[m]
        else:
            p_power[n] = p
            sigma_p[n] = 1 + p
            m = n // p
            sigma[n] = sigma_p[n] * sigma[m]
    
    return sigma

def _ratio_err(a, b):
    """Return |a/b - 3/2| as a Fraction."""
    num = abs(2 * a - 3 * b)
    if num == 0:
        return None  # exact match, invalid
    den = 2 * b
    return Fraction(num, den)

def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin before 90s kill
    
    N = MAX_N
    
    # Build SPF table
    spf = _sieve_spf(N)
    
    if time.time() > deadline:
        return [11, 6]
    
    # Compute sigma for all numbers up to N
    sigma = _sigma_sieve(N, spf)
    
    if time.time() > deadline:
        return [11, 6]
    
    # Group numbers by their sigma value
    groups = defaultdict(list)
    for i in range(1, N + 1):
        groups[sigma[i]].append(i)
    
    if time.time() > deadline:
        return [11, 6]
    
    # Filter to groups with at least 2 elements and sort them
    for s in list(groups.keys()):
        if len(groups[s]) >= 2:
            groups[s].sort()
        else:
            del groups[s]
    
    if time.time() > deadline:
        return [11, 6]
    
    # Search for best pair (a, b) with sigma(a) == sigma(b) and |a/b - 3/2| minimized
    # a/b close to 3/2 means a ≈ 1.5 * b
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # For each group, for each element b, binary search for a close to 1.5*b
    
    best_error = Fraction(1, 1)  # start with a large error
    best_pair = (11, 6)  # fallback
    
    for s, arr in groups.items():
        if time.time() > deadline:
            break
        
        # For each b in arr, find a in arr closest to 1.5*b
        # We want a > b typically (since 3/2 > 1)
        for i, b in enumerate(arr):
            if time.time() > deadline:
                break
            
            target_a = 3 * b / 2.0
            pos = bisect.bisect_left(arr, int(target_a))
            
            # Check neighbors around the insertion point
            candidates = set()
            for offset in [-2, -1, 0, 1, 2]:
                idx = pos + offset
                if 0 <= idx < len(arr):
                    candidates.add(arr[idx])
            
            for a in candidates:
                if a == b:
                    continue
                err = _ratio_err(a, b)
                if err is not None and err < best_error:
                    best_error = err
                    best_pair = (a, b)
    
    a, b = best_pair
    
    # Verify sigma(a) == sigma(b)
    if sigma[a] != sigma[b]:
        return [11, 6]
    
    # Return as list of Python ints
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
