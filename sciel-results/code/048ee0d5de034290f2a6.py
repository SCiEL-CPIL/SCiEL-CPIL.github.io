import math
import time
from functools import lru_cache

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0  # Safety margin
    
    # Candidates must share a factor with 255 (3, 5, 17)
    # 255 is always included.
    # We find the max clique in the graph of other candidates U.
    
    # 1. Generate candidates U = {x in [1, 254] | gcd(x, 255) > 1}
    candidates = []
    for x in range(1, N):
        if math.gcd(x, N) > 1:
            candidates.append(x)
            
    n_cands = len(candidates)
    
    # 2. Build adjacency bitsets using prime factorization for speed
    # Factor each candidate into primes
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
    
    # Precompute prime factors for each candidate
    prime_factors = [factorize(candidates[i]) for i in range(n_cands)]
    
    # Build adjacency: adj[i] is bitmask of j where candidates[i] and candidates[j] share a factor
    adj = [0] * n_cands
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            if prime_factors[i] & prime_factors[j]:  # non-empty intersection
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 3. Find Maximum Clique using optimized Bron-Kerbosch with Pivoting
    max_clique_size = 0
    max_clique_set = []
    
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
        # Fallback: Multiples of 3
        result = [x for x in range(1, N + 1) if x % 3 == 0]
        
    if N not in result:
        result.append(N)
    
    return result
