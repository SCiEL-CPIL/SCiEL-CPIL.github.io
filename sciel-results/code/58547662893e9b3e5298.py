"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import defaultdict
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sigma_sieve(n):
    """Compute sigma(k) for k=1..n using a sieve."""
    sigma = [0] * (n + 1)
    for d in range(1, n + 1):
        for m in range(d, n + 1, d):
            sigma[m] += d
    return sigma

def construct():
    t_start = time.time()
    deadline = t_start + 55.0  # safety margin before 60s kill (we have 90s but be safe)
    
    N = 10_000_000
    target_frac = Fraction(3, 2)
    
    # Compute sigma for all numbers up to N
    sigma = _sigma_sieve(N)
    
    # Group numbers by their sigma value
    # Only keep sigma values that appear at least twice (potential pairs)
    # But first, let's just group everything
    groups = defaultdict(list)
    for i in range(1, N + 1):
        groups[sigma[i]].append(i)
    
    # Filter to groups with at least 2 elements
    # Sort each group
    for s in list(groups.keys()):
        if len(groups[s]) >= 2:
            groups[s].sort()
        else:
            del groups[s]
    
    # Search for best pair (a, b) with sigma(a) == sigma(b) and |a/b - 3/2| minimized
    # a/b close to 3/2 means a ≈ 1.5 * b, so a > b typically (or b > a if we consider a/b < 3/2)
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # So for a given b, we want a close to 3b/2.
    # For each group, for each b in the group, binary search for a close to 3b/2.
    
    best_error = Fraction(1, 1)  # start with a large error
    best_pair = (1, 1)  # placeholder, will be replaced
    
    # To be efficient, iterate over groups. For each group, for each element b,
    # look for the element a in the same group closest to 3b/2.
    # Since the group is sorted, we can binary search.
    
    import bisect
    
    for s, arr in groups.items():
        if time.time() > deadline:
            break
        # For each b in arr, find a in arr closest to 3b/2
        # a should be such that a/b ≈ 3/2, i.e., a ≈ 1.5*b
        # We can iterate over b and binary search for 1.5*b in arr
        # To avoid checking too many, we can sample or limit, but let's try all first
        # with a time check
        for i, b in enumerate(arr):
            if time.time() > deadline:
                break
            target_a = 3 * b / 2.0
            # Binary search for target_a in arr
            pos = bisect.bisect_left(arr, int(target_a))
            # Check pos and pos-1
            candidates = []
            if pos < len(arr):
                candidates.append(arr[pos])
            if pos > 0:
                candidates.append(arr[pos - 1])
            # Also check pos+1 in case of floating point issues
            if pos + 1 < len(arr):
                candidates.append(arr[pos + 1])
            if pos - 2 >= 0:
                candidates.append(arr[pos - 2])
            
            for a in set(candidates):
                if a == b:
                    continue
                # Compute |a/b - 3/2| exactly as a Fraction
                # a/b - 3/2 = (2a - 3b) / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num == 0:
                    # This would be an exact match, which is invalid per the task
                    # "A raw value of exactly 0 cannot be scored and counts as invalid"
                    # So we skip this
                    continue
                err = Fraction(num, den)
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)
    
    # Ensure we have a valid pair
    if best_pair == (1, 1):
        # Fallback: find any valid pair
        for s, arr in groups.items():
            if len(arr) >= 2:
                a, b = arr[0], arr[1]
                if a != b:
                    best_pair = (a, b)
                    break
        if best_pair == (1, 1):
            # Last resort
            best_pair = (11, 6)
    
    a, b = best_pair
    # Verify sigma(a) == sigma(b)
    # (should be true by construction, but just in case)
    # If not, fall back
    if sigma[a] != sigma[b]:
        best_pair = (11, 6)
        a, b = best_pair
    
    # Return as list of Python ints
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
