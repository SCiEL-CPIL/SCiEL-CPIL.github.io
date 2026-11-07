import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s (90s kill)
    
    n = 256
    universe = list(range(2, n + 1))
    N = n + 1  # index 0,1 unused; 2..256 used
    
    # Precompute: for each a, the set of (b,c) pairs where a | b*c
    # But we need efficient checking. Instead, precompute for each a, 
    # a bitmask of "conflict" - but conflicts are on pairs.
    
    # Alternative: for a given set S, check if valid by iterating triples.
    # For local search, we need fast "can I add x" and "can I remove x" checks.
    
    # Key insight: a | b*c iff there exist b,c in S\{a} such that a divides b*c.
    # For a fixed a and set S, this is: does there exist b in S, c in S, b!=c, b!=a, c!=a with a|bc?
    
    # Precompute for each a: list of b values where gcd(a,b) > 1 might help,
    # but actually a|bc is more complex.
    
    # Let's precompute: div_pairs[a] = set of (b,c) with b<c, b,c in 2..256, b!=c, a|b*c
    # This is O(N^3) = ~256^3 = 16M which is fine.
    
    # Actually, for each a, for each b, we need c such that a | b*c.
    # a | b*c means c must be a multiple of a/gcd(a,b).
    # Let g = gcd(a,b), then a/g | c. So c must be a multiple of a/g.
    
    from math import gcd
    
    # Precompute conflict_mask[a] as a set of b values that "participate" in a conflict
    # Actually, let's think differently.
    
    # For local search, the most efficient approach:
    # - Maintain S as a set
    # - To check if x can be added: for all a in S, check if a | x*c for some c in S (c!=a, c!=x)
    #   AND for all a,b in S, check if a | b*x (i.e., x is the "c" in a|b*c)
    #   AND x | a*b for some a,b in S
    
    # This is O(|S|^2) per check, which with |S|~100+ and many iterations could be slow.
    
    # Better: precompute for each pair (a,b) the set of c such that a | b*c.
    # Then for a given S, a conflict exists iff there's (a,b,c) in S^3 distinct with c in conflict_set[a][b].
    
    # Let me precompute: for each a, for each b != a, the set of c != a,b such that a | b*c.
    # conflict_c[a][b] = frozenset of c values
    
    # This is O(N^2 * N) = O(N^3) memory in worst case, but we can store as bitmasks.
    
    # Use bitmasks: for each a, b, store a bitmask of c values where a | b*c.
    # 256 bits = 32 bytes, 255*254 = ~65000 pairs, so ~2MB. Fine.
    
    # Actually, let's use a different approach for speed:
    # Precompute for each a: a list of (b, c_mask) where c_mask is a bitmask of c such that a|b*c
    # But this is still large.
    
    # Simpler: just do O(|S|^2) checks but optimize with early termination and numpy.
    
    # Let me try a different strategy: 
    # 1. Start with all primes (they form a valid set? No, because if p is prime and 2p is in set, 
    #    then for a=p, b=2p, c=anything: p | 2p*c, so conflict!)
    #    So primes and their multiples can't coexist.
    
    # 2. Better initial set: numbers that are "large" (harder to divide products)
    
    # Let me just do a good local search from multiple starts.
    
    def is_valid_set(S):
        """Check if S is valid. S is a set of ints."""
        S_list = list(S)
        L = len(S_list)
        for i in range(L):
            a = S_list[i]
            for j in range(L):
                if j == i: continue
                b = S_list[j]
                for k in range(L):
                    if k == i or k == j: continue
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to S."""
        S_list = list(S)
        # Check: for all distinct a,b in S, a | b*x? (x plays role of c)
        # And x | a*b? (x plays role of a)
        # And for a in S: a | x*c for some c in S? (x plays role of b)
        
        for a in S_list:
            # x | a*b for some b in S, b != a
            for b in S_list:
                if b == a: continue
                if (a * b) % x == 0:
                    return False
            
            # a | x*c for some c in S, c != a
            for c in S_list:
                if c == a: continue
                if (x * c) % a == 0:
                    return False
            
            # a | b*x for some b in S, b != a
            for b in S_list:
                if b == a: continue
                if (b * x) % a == 0:
                    return False
        
        return True
    
    def can_remove(S, x):
        """Always true - removing never creates conflicts."""
        return True
    
    def try_add(S, x):
        """Add x to S if possible. Returns new S or None."""
        if x in S:
            return S
        if can_add(S, x):
            S.add(x)
            return S
        return None
    
    # Greedy construction with different orderings
    def greedy(order):
        S = set()
        for x in order:
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        return S
    
    # Local search: try to add elements, then try swaps
    def local_search(S, max_time=None):
        S = set(S)
        improved = True
        while improved and (max_time is None or time.time() < max_time):
            improved = False
            others = [x for x in range(2, n + 1) if x not in S]
            random.shuffle(others)
            
            # Try additions
            for x in others:
                if time.time() >= deadline - 5:
                    return S
                if can_add(S, x):
                    S.add(x)
                    improved = True
            
            # Try swaps: remove one, add one
            S_list = list(S)
            random.shuffle(S_list)
            for a in S_list:
                if time.time() >= deadline - 5:
                    return S
                S.remove(a)
                for x in others:
                    if x in S:
                        continue
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        break
                else:
                    S.add(a)  # put it back
            
        return S
    
    # Multiple random restarts with different initial sets
    best_S = set()
    
    # Start 1: all primes
    primes = [a for a in range(2, n + 1) if all(a % d for d in range(2, int(a**0.5) + 1))]
    S1 = greedy(primes)
    S1 = local_search(S1)
    if len(S1) > len(best_S):
        best_S = set(S1)
    
    # Start 2: all numbers descending (larger numbers first)
    S2 = greedy(range(n, 1, -1))
    S2 = local_search(S2)
    if len(S2) > len(best_S):
        best_S = set(S2)
    
    # Start 3: all numbers ascending
    S3 = greedy(range(2, n + 1))
    S3 = local_search(S3)
    if len(S3) > len(best_S):
        best_S = set(S3)
    
    # Start 4: random order
    for _ in range(5):
        order = list(range(2, n + 1))
        random.shuffle(order)
        S = greedy(order)
        S = local_search(S)
        if len(S) > len(best_S):
            best_S = set(S)
    
    # Final local search on best
    best_S = local_search(best_S)
    
    # Verify
    assert is_valid_set(best_S), "Invalid set returned!"
    
    return list(best_S)
