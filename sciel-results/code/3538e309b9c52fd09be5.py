import time
from collections import defaultdict
import math

def construct():
    N = 10**7
    t0 = time.time()
    deadline = 82.0  # safety margin before 90s kill
    
    # Step 1: Compute sigma(n) for n = 1..N using sieve
    # sigma(n) = sum of divisors of n
    # Using the divisor-sum sieve: for each i, add i to sigma[j] for all j multiple of i
    # This is O(N log N) which for N=10^7 is about 10^7 * 16 ≈ 1.6 * 10^8 operations
    # In Python this might be too slow. Let's use a more efficient approach.
    
    # Actually, let's use the fact that sigma is multiplicative and compute via factorization.
    # But for N=10^7, we need a fast method.
    
    # Best approach: use the standard sieve method with numpy for speed.
    # But we need to be careful about time.
    
    # Alternative: use a pure Python sieve but optimized.
    # sigma[n] starts at 0, then for d from 1 to N, for m from d to N step d, sigma[m] += d
    # This is O(N log N).
    
    # For N = 10^7, N log N ≈ 10^7 * 16 = 1.6 * 10^8 additions.
    # In Python, each iteration is ~50ns, so 1.6*10^8 * 50ns = 8 seconds. That's tight but doable.
    # But actually it's more like 100-200ns per operation in a tight loop.
    # Let's try to use array module or just a list.
    
    # Actually, let me reconsider. The total number of operations is sum_{d=1}^{N} N/d = N * H(N) ≈ N * ln(N) ≈ 10^7 * 16.1 = 1.61 * 10^8.
    # In a tight Python loop, this might take 10-20 seconds. Let's try.
    
    # We'll use a list of zeros.
    sigma = [0] * (N + 1)
    
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            sigma[m] += d
    
    if time.time() > deadline:
        return [11, 6]
    
    # Step 2: Build index: sigma_value -> list of n
    # We only need groups where at least 2 numbers share the same sigma.
    sigma_to_ns = defaultdict(list)
    for n in range(1, N + 1):
        s = sigma[n]
        sigma_to_ns[s].append(n)
    
    if time.time() > deadline:
        return [11, 6]
    
    # Step 3: For each group, find the pair (a, b) that minimizes |a/b - 3/2|
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # We want to minimize this.
    
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 3/2)  # = |11/6 - 9/6| = 2/6 = 1/3 ≈ 0.333
    
    # For each sigma group, we have a set of numbers.
    # We want to find a, b in the group such that |2a - 3b| / (2b) is minimized.
    # Note: a and b are distinct? The problem says "a ratio" and sigma(a)=sigma(b). 
    # If a == b, then a/b = 1, and |1 - 3/2| = 1/2. That's worse than 1/3.
    # So we want a ≠ b presumably, but let's not assume.
    
    # For a given group G, to minimize |2a - 3b|/(2b):
    # For each b in G, the best a is the one in G closest to 3b/2.
    # So for each b, find a in G such that a ≈ 3b/2, minimizing |2a - 3b|.
    
    # If we sort each group, we can use binary search for each b.
    # But groups might be large. Let's see how many groups there are and their sizes.
    
    # Number of unique sigma values: for n up to 10^7, sigma(n) can be up to about 2.4*10^7.
    # The number of distinct sigma values is much less than N.
    
    # Strategy: for each group, sort it. Then for each b in the group, binary search for 3b/2.
    # Check the nearest elements.
    
    # But iterating over all groups and all elements might be slow if groups are large.
    # Let's estimate: total elements across all groups = N = 10^7.
    # If we do a binary search (O(log |G|)) for each element, total is O(N log(max_group_size)).
    # max_group_size is probably not too large (maybe a few hundred?).
    # So total is about 10^7 * 10 = 10^8 operations. Should be fine.
    
    # Let's do it.
    
    import bisect
    
    for s_val, group in sigma_to_ns.items():
        if len(group) < 2:
            continue
        
        group_sorted = sorted(group)
        
        # For each b in group, find best a
        for b in group:
            target = 1.5 * b
            # Binary search for target in group_sorted
            idx = bisect.bisect_left(group_sorted, target)
            
            # Check candidates: idx-1, idx, idx+1 (within bounds)
            candidates = []
            for i in [idx - 1, idx, idx + 1]:
                if 0 <= i < len(group_sorted):
                    candidates.append(group_sorted[i])
            
            for a in candidates:
                if a == b:
                    # a/b = 1, error = 1/2. Skip unless it's the only option.
                    # Actually, let's include it but it won't beat 1/3.
                    pass
                err = abs(2*a - 3*b) / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
        
        if time.time() > deadline:
            break
    
    # Ensure a ≠ b? The problem doesn't explicitly forbid a=b, but "ratio" and the 
    # structure suggests they should be different. Also |a/b - 3/2| = 1/2 if a=b,
    # which is worse than our current best of 1/3. So it's fine.
    
    # Also need a, b to be positive integers <= N, which they are by construction.
    
    # Edge case: best_err might still be 1/3 if no better pair found.
    # Return as Python ints.
    
    return [int(best_a), int(best_b)]
