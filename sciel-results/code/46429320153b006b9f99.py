def construct():
    import time
    from math import gcd
    
    deadline = time.time() + 85  # 85s to be safe before 90s kill
    
    N = 512
    
    # Precompute LCM compatibility: compatible[i][j] = lcm(i,j) <= N
    # Use a bitset approach for efficiency
    # First, let's try a greedy approach: sort by "constraint" and greedily add
    
    # Key insight: if all numbers in our set divide some M <= 512, then any pair has LCM dividing M, so LCM <= M <= 512.
    # So one valid set is all divisors of 512. But we can do better.
    
    # Better: use the fact that if we pick numbers all of the form 2^a * p^b where p is a fixed odd prime,
    # and 2^a * p^b <= 512, then lcm of two such numbers divides 2^max(a1,a2) * p^max(b1,b2) <= 512 if we're careful.
    
    # Actually, simplest strong approach: greedy with backtracking-like heuristic
    # Sort candidates by "how many conflicts they cause" - add least-conflicting first
    
    # Precompute for each i, the set of j > i where lcm(i,j) > 512
    # Use numpy for speed if available, else pure Python
    
    import numpy as np
    
    # Build conflict matrix using vectorized operations where possible
    # For each i, compute lcm(i, j) for all j efficiently
    
    # lcm(i,j) = i * j // gcd(i,j)
    # So lcm(i,j) > 512 iff i * j // gcd(i,j) > 512
    
    # Let's use a different strategy: 
    # The maximum set likely includes many powers of 2 and numbers with small odd parts.
    
    # Greedy: start with all numbers, iteratively remove the one that causes the most conflicts
    # But that's O(n^3) which might be too slow for n=512
    
    # Better greedy: sort by number of compatible partners (ascending), add in that order if compatible with all selected
    
    # Compute compatibility counts
    comp_count = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            lcm_val = i * j // g
            if lcm_val <= N:
                comp_count[i] += 1
                comp_count[j] += 1
    
    # Sort by comp_count ascending (most constrained first)
    candidates = list(range(1, N + 1))
    candidates.sort(key=lambda x: (comp_count[x], x))
    
    selected = []
    selected_set = set()
    
    for c in candidates:
        # Check if c is compatible with all in selected
        ok = True
        for s in selected:
            g = gcd(c, s)
            if c * s // g > N:
                ok = False
                break
        if ok:
            selected.append(c)
            selected_set.add(c)
        
        if time.time() > deadline:
            break
    
    # Try to add more numbers that weren't in candidates order
    # Actually, let's also try a different greedy: sort by value
    
    # Alternative: try to include all powers of 2 and numbers of form 2^a * 3, 2^a * 5, etc.
    
    # Let's also try: for each odd number m <= 512, include all 2^a * m <= 512
    # But only if m1 * m2 <= 512 for any two odd parts m1, m2 in our set (since lcm(2^a*m1, 2^b*m2) = 2^max(a,b) * lcm(m1,m2) and m1,m2 odd so lcm(m1,m2) >= m1*m2/gcd(m1,m2))
    
    # This is getting complex. Let's stick with the greedy and see what size we get.
    
    if not selected:
        return list(range(1, 11))  # fallback: divisors of 512 up to 512
    
    return selected
