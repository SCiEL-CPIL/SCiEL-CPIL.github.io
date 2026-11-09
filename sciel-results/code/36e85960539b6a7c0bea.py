import math

def construct():
    # Limit for generating powerful numbers
    # 10^9 allows ~20,000 powerful numbers, ~2*10^8 pairs, feasible in ~10-20s
    L = 10**9
    
    # Generate powerful numbers up to L
    # A number is powerful if it is of the form u^2 * v^3 where v is square-free
    powerful = set()
    
    # Precompute primes up to cube root of L for square-free check?
    # Actually, we just need to check if v is square-free.
    # v goes up to L^(1/3) ~ 1000.
    # Simple trial division is fine.
    
    max_v = int(L ** (1/3)) + 2
    
    for v in range(1, max_v + 1):
        # Check if v is square-free
        is_sf = True
        temp = v
        p = 2
        while p * p <= temp:
            if temp % (p * p) == 0:
                is_sf = False
                break
            p += 1 if p == 2 else 2
        if not is_sf:
            continue
        
        # u^2 * v^3 <= L
        # u <= sqrt(L / v^3)
        max_u = int(math.isqrt(L // (v ** 3)))
        for u in range(1, max_u + 1):
            powerful.add(u * u * (v ** 3))
    
    P = sorted(powerful)
    n_p = len(P)
    
    # Count sums
    counts = {}
    # To speed up, we can use a list if the range is small, but range is 2*L = 2*10^9, too big.
    # Dictionary is necessary.
    for i in range(n_p):
        ai = P[i]
        # We can break inner loop if ai + P[j] is getting too large?
        # No, we want the global max.
        for j in range(i, n_p):
            s = ai + P[j]
            counts[s] = counts.get(s, 0) + 1
            
    # Find the N0 with the most representations
    best_n0 = max(counts, key=counts.get)
    best_count = counts[best_n0]
    
    # Retrieve the pairs for best_n0
    p_set = set(P)
    pairs = []
    for a in P:
        b = best_n0 - a
        if a <= b and b in p_set:
            pairs.append([a, b])
            
    # Scale to near 10^18
    target = 10**18
    c = int(math.isqrt(target // best_n0))
    
    # Ensure N <= 10^18
    N = c * c * best_n0
    if N > target:
        c -= 1
        N = c * c * best_n0
        
    scaled_pairs = [[c * c * a, c * c * b] for a, b in pairs]
    
    return {'N': N, 'pairs': scaled_pairs}
