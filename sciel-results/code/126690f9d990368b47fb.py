"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    # Powerful number = a^2 * b^3 where b is square-free
    # b^3 <= limit => b <= limit^(1/3)
    max_b = int(limit ** (1.0/3.0)) + 2
    if max_b < 1:
        max_b = 1
    
    # Sieve for square-free numbers up to max_b
    is_sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        is_sq_free[0] = False
    if max_b >= 1:
        is_sq_free[1] = True
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        for j in range(sq, max_b + 1, sq):
            is_sq_free[j] = False
    
    powerful = set()
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b * b * b
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def _is_powerful(n):
    """Check if n is powerful by exact factorization."""
    if n == 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count == 1:
                return False
        p += 1 if p == 2 else 2
    # temp is now a prime factor
    if temp > 1:
        # It appears exactly once
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    N_LIMIT = 10**18
    
    # Generate powerful numbers up to N_LIMIT
    powerful = _generate_powerful(N_LIMIT)
    powerful_set = set(powerful)
    n_powerful = len(powerful)
    
    elapsed = time.time() - start_time
    if time.time() > deadline:
        # Fallback: return a simple valid result
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Strategy: Find N that maximizes count of (a, b) with a+b=N, a,b powerful
    # Use the fact that for a fixed N, we need a in powerful and N-a in powerful
    # 
    # Key insight: If we pick N = a + b for many pairs, we want N to be a "hot spot"
    # 
    # Approach: For each powerful number a, consider N = a + b for b in powerful
    # But that's O(M^2) which is too slow for M ~ 10^6
    # 
    # Better: Use the structure. Powerful numbers have the form a^2 * b^3.
    # Let's try specific families of N.
    #
    # Alternative: Pick N and count. We can be smart about which N to try.
    #
    # Strategy 2: For each powerful a, N = a + a = 2a is always valid (a, a)
    # But we want distinct pairs a <= b, so a=a gives one pair.
    #
    # Strategy 3: Use the fact that 1 is powerful. So N = 1 + a for any powerful a.
    # That gives one representation per powerful a.
    #
    # Better strategy: Find N where many powerful numbers are symmetric around N/2.
    #
    # Let's try: For each powerful number a, N = a + b where b is also powerful.
    # We can iterate over a in powerful and for each, check a few b values.
    #
    # Most efficient: Pick candidate N values and count representations.
    #
    # Candidate N generation:
    # - N = 2*a for powerful a (gives at least 1 pair: (a,a))
    # - N = a + b for small powerful a, b
    # - N = sum of two specific powerful numbers
    #
    # Let's try a different approach: 
    # For each powerful number a, N = a + b where b ranges over powerful.
    # Count occurrences of each N.
    # But M^2 is too large.
    #
    # Instead, let's use the structure more directly.
    # Powerful numbers: 1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100, 108, 121, 125, 128, 144, 169, 196, 200, 216, 225, 243, 256, 289, 324, 343, 361, 400, 441, 484, 500, 512, 529, 576, 625, 675, 676, 729, 784, 841, 864, 900, 961, 1000, 1024, 1089, 1152, 1156, 1225, 1296, 1331, 1369, 1444, 1521, 1568, 1600, 1681, 1728, 1764, 1849, 1936, 2025, 2048, 2116, 2187, 2209, 2304, 2401, 2500, 2601, 2704, 2744, 2809, 2916, 3025, 3125, 3136, 3249, 3364, 3481, 3600, 3721, 3844, 3969, 4096, 4225, 4356, 4489, 4624, 4761, 4900, 5041, 5184, 5329, 5476, 5625, 5776, 5929, 6084, 6241, 6400, 6561, 6724, 6889, 7056, 7225, 7396, 7569, 7744, 7921, 8100, 8281, 8464, 8649, 8836, 9025, 9216, 9409, 9604, 9801, 10000, ...
    #
    # Let me try a computational approach with a time budget.
    #
    # Phase 1: Try N = 2*a for all powerful a, count pairs.
    # Phase 2: For promising N, do a full count.
    #
    # Actually, the best approach given time constraints:
    # 1. Generate powerful numbers (done)
    # 2. For a subset of candidate N values, count representations
    # 3. Candidate N: sums of pairs of powerful numbers where both are "small" relative to N
    #
    # Let's try: N = a + b where a, b are powerful and a <= b.
    # We'll iterate over a in powerful (up to some limit) and b in powerful (>= a)
    # and count N values. But this is still O(M^2).
    #
    # Alternative: Use numpy for vectorized counting.
    #
    # Let's try a different structural approach:
    # Pick N = k * m where k, m have special properties.
    #
    # Actually, let's just do a focused search:
    # For each powerful a (up to a limit), consider N = a + b for b in powerful.
    # Use a dictionary to count N values.
    # Limit the inner loop to keep it fast.
    
    import numpy as np
    
    # Convert to numpy array for speed
    pow_arr = np.array(powerful, dtype=np.int64)
    
    # Strategy: For each a in a subset of powerful, compute N = a + b for b in powerful
    # and count frequencies. Use the first K powerful numbers for a, and all for b.
    # This is O(K * M) which for K=1000, M=10^6 is 10^9 - too slow.
    #
    # Better: Use convolution-like approach or just try specific N candidates.
    #
    # Let's try: N = a + b where a and b are both from a small set of powerful numbers.
    # The most "productive" powerful numbers are those that can pair with many others.
    #
    # Key insight: 1 is powerful. So for any powerful a, N = 1 + a is valid.
    # Also 4 is powerful, so N = 4 + a is valid for any powerful a.
    #
    # Let's try N values that are sums involving small powerful numbers.
    # For small powerful s, N = s + a for a in powerful gives many representations
    # if a can be expressed as N - s in many ways... no, that's just one way per a.
    #
    # Wait, I need to count (a, b) pairs with a + b = N.
    # For a fixed N, the number of representations is the number of a in powerful 
    # such that N - a is also in powerful and a <= N - a.
    #
    # So I need to find N that maximizes |{a in powerful : a <= N/2 and N-a in powerful}|.
    #
    # Let's use a smarter search. Try N values that are likely to be "central" 
    # in the powerful number distribution.
    #
    # Approach: For a range of N values, count representations using set membership.
    # Pick N values spaced out, then refine.
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Try N = 2*a for powerful a (guaranteed at least 1 pair)
    # Also try N = a + b for small a, b
    
    # Phase 1: Quick scan of candidate N values
    # Candidate N: 2*a for a in powerful (up to N_LIMIT/2)
    # Also: a + b for small powerful a, b
    
    # Let's try a focused approach:
    # For each powerful a up to some limit, N = a + a = 2a
    # Count how many powerful b satisfy b = 2a - a = a... just one.
    # But also b could be other values if 2a - b is powerful.
    
    # Let me just do a direct search with a time budget.
    # For candidate N values, count representations.
    
    # Generate candidate N values:
    # 1. N = 2*a for a in powerful (a <= N_LIMIT//2)
    # 2. N = a + b for a, b in first K powerful numbers
    # 3. N = a + 1, a + 4, a + 8, ... for a in powerful
    
    candidates = set()
    
    # Add 2*a for powerful a
    for a in powerful:
        if a > N_LIMIT // 2:
            break
        candidates.add(2 * a)
    
    # Add a + b for small a, b (first 1000 powerful numbers)
    small_pow = powerful[:min(2000, len(powerful))]
    for i, a in enumerate(small_pow):
        for b in small_pow[i:]:
            n_val = a + b
            if n_val <= N_LIMIT:
                candidates.add(n_val)
            else:
                break
    
    # Add a + s for small powerful s
    small_s = powerful[:min(100, len(powerful))]
    for a in powerful:
        if a > N_LIMIT:
            break
        for s in small_s:
            n_val = a + s
            if n_val <= N_LIMIT:
                candidates.add(n_val)
    
    candidates = sorted(candidates)
    
    # Now evaluate each candidate N
    # Count representations: number of a in powerful with a <= N/2 and N-a in powerful_set
    
    # To speed up, pre-filter: only consider N where N/2 is near many powerful numbers
    
    for N in candidates:
        if time.time() > deadline:
            break
        if N < 2 or N > N_LIMIT:
            continue
        
        # Count representations
        count = 0
        pairs = []
        half = N // 2
        # Iterate over powerful a <= half
        for a in powerful:
            if a > half:
                break
            b = N - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
