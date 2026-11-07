import time
import numpy as np

def _sigma_sieve(N):
    """Compute sigma(n) for all n up to N using a divisor-sum sieve."""
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
    return sigma

def construct():
    deadline = time.time() + 75.0  # safety margin before 90s kill
    MAX_N = 10_000_000
    
    # Step 1: Precompute sigma for all n up to MAX_N
    sigma = _sigma_sieve(MAX_N)
    
    # Step 2: Group indices by sigma value, then for each group find pairs with ratio close to 3/2
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For a fixed b, the best a is the one in the same sigma group closest to 3b/2.
    
    # Build a mapping from sigma value to sorted list of n values
    # This is memory-intensive for 10^7, so let's be smarter.
    
    # Alternative: for each b, compute target_a = 3b/2, and look for a in the same sigma group
    # closest to target_a. We can use a dictionary mapping sigma values to sorted arrays.
    
    # But building sorted arrays for all sigma values up to 10^7 is expensive.
    # Let's instead iterate and use a different approach.
    
    # Approach: For each possible sigma value s, collect all n with sigma(n) = s.
    # Then for each pair (a, b) in the same group, check if |2a - 3b| / (2b) is small.
    
    # More efficient: for each b, the ideal a is 3b/2. We want to find a in the same sigma group
    # as b that is closest to 3b/2.
    
    # Let's build the groups. We'll use a dictionary.
    # To save memory, we can process in chunks or use a different strategy.
    
    # Actually, let's just build the full mapping. There are at most 10^7 distinct sigma values,
    # but in practice far fewer. Each n contributes to exactly one group.
    
    from collections import defaultdict
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        groups[int(sigma[n])].append(n)
    
    # Now for each group, sort the list and for each element b, binary search for the element
    # closest to 3b/2.
    
    best_err = float('inf')
    best_a, best_b = 11, 6  # fallback: sigma(11)=12, sigma(6)=12, |11/6 - 3/2| = 1/3
    
    import bisect
    
    for s, lst in groups.items():
        if time.time() > deadline - 15:
            break
        if len(lst) < 2:
            continue
        lst.sort()
        for i, b in enumerate(lst):
            target = 3 * b / 2.0
            # Binary search for target in lst
            pos = bisect.bisect_left(lst, target)
            # Check pos and pos-1
            for j in (pos - 1, pos):
                if 0 <= j < len(lst):
                    a = lst[j]
                    if a == b:
                        continue
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue  # exact 3/2, invalid
                    den = 2 * b
                    err = num / den
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
    
    return [int(best_a), int(best_b)]
