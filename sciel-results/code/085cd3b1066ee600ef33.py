import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0  # Safety margin before 90s kill
    
    # Key insight: 255 = 3*5*17
    # All elements in our set must share a factor with 255.
    # The largest obvious clique is all multiples of 3 (85 elements including 255).
    # We verify this is optimal via Bron-Kerbosch if time permits.
    
    # Quick baseline: multiples of 3
    base_set = [x for x in range(1, N + 1) if x % 3 == 0]
    best_size = len(base_set)
    best_set = base_set
    
    # Generate candidates: x in [1, 254] with gcd(x, 255) > 1
    candidates = []
    for x in range(1, N):
        if math.gcd(x, N) > 1:
            candidates.append(x)
    
    n_cands = len(candidates)
    
    if n_cands == 0:
        return best_set
    
    # Build adjacency using prime factor sets
    def factorize(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    prime_factors = [factorize(candidates[i]) for i in range(n_cands)]
    
    # Build adjacency bitsets
    adj = [0] * n_cands
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            if prime_factors[i] & prime_factors[j]:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Bron-Kerbosch with pivoting
    max_clique_size = best_size  # Start with our known good solution
    max_clique_set = base_set[:]
    
    def popcount(x):
        return bin(x).count('1')
    
    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique_size, max_clique_set
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                max_clique_set = R[:]
            return
        
        # Pruning: if len(R) + popcount(P) <= max_clique_size, can't beat current best
        if len(R) + popcount(P) <= max_clique_size:
            return
        
        # Pivot selection: choose u in P union X with max degree in P
        PX = P | X
        if PX == 0:
            return
        
        best_u = -1
        max_deg = -1
        for u in iter_bits(PX):
            deg = popcount(P & adj[u])
            if deg > max_deg:
                max_deg = deg
                best_u = u
        
        if best_u == -1:
            return

        # Recurse on P \ N(u)
        P_not_N_u = P & ~adj[best_u]
        
        for v in iter_bits(P_not_N_u):
            R.append(candidates[v])
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bron_kerbosch(R, P_new, X_new)
            
            R.pop()
            P &= ~(1 << v)
            X |= (1 << v)
    
    # Initial call
    initial_P = (1 << n_cands) - 1
    bron_kerbosch([], initial_P, 0)
    
    # Construct the final set
    if max_clique_set:
        result = max_clique_set + [N]
    else:
        result = base_set
        
    # Ensure 255 is included and no duplicates
    if N not in result:
        result.append(N)
    result = list(set(result))
    
    return result
