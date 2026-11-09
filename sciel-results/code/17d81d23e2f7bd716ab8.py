"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from math import isqrt

def _build_hypergraph(n=256):
    """
    Build a 3-uniform hypergraph on vertices 2..n.
    Hyperedge {a,b,c} exists if any of a|bc, b|ac, c|ab holds (with a,b,c distinct).
    We want maximum independent set: no hyperedge fully contained in the set.
    Equivalently, for every triple of distinct members a,b,c in the set,
    a does NOT divide b*c, b does NOT divide a*c, c does NOT divide a*b.
    """
    vertices = list(range(2, n + 1))
    # For efficiency, precompute for each pair (b,c) the set of a that divide b*c
    # Actually, let's think about it differently.
    # We need to check: for all distinct a,b,c in S, a ∤ b*c.
    # This is a 3-uniform constraint.
    
    # Let's build adjacency: for each vertex a, which pairs (b,c) are "forbidden" with a?
    # i.e., a | b*c.
    
    # For branch and bound, we need to be able to quickly check if adding a vertex
    # creates a forbidden triple.
    
    # Precompute: for each a, the set of b such that there exists c with a|b*c and b,c distinct from a and each other.
    # Actually, let's precompute a 3D boolean or use a different approach.
    
    # Let's precompute divisors. For each number x in 2..n*n (since b*c can be up to 256*255),
    # but that's large. Instead, for each a, we can compute which b values have a|b*c for some c.
    
    # Alternative: precompute for each pair (b,c), the set of divisors of b*c that are in 2..n.
    # Then a hyperedge {a,b,c} exists if a is in divisors(b*c) ∩ [2,n] and a ≠ b, a ≠ c.
    
    # Let's precompute divisors for all products b*c where b,c in 2..n, b≠c.
    # That's about 255*254/2 ≈ 32K products, each needing factorization/divisor enumeration.
    
    # Actually, for the branch and bound, we need a fast check. Let's precompute:
    # forbid[a][b] = set of c such that a | b*c (with a,b,c distinct)
    # This is a 2D list of sets.
    
    forbid = [set() for _ in range(n + 1)]
    for a in range(2, n + 1):
        for b in range(2, n + 1):
            if b == a:
                continue
            # Find all c in 2..n, c != a, c != b, such that a | b*c
            # a | b*c iff (a / gcd(a,b)) | c
            g = _gcd(a, b)
            need = a // g
            if need > n:
                continue
            # c must be a multiple of need, c in 2..n, c != a, c != b
            for c in range(need, n + 1, need):
                if c != a and c != b:
                    forbid[a][b].add(c)
    return vertices, forbid

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Precompute the hypergraph data
_VERTICES, _FORBID = _build_hypergraph(256)

def construct():
    """
    Find maximum independent set in the 3-uniform hypergraph.
    Use branch and bound.
    """
    n = 256
    # Vertices are 2..256
    # We'll use a recursive branch and bound.
    # State: current set S, remaining candidates R.
    # Upper bound: len(S) + len(R)
    # But we can do better: if we can quickly determine that some vertices in R
    # cannot be added without creating a forbidden triple, we can prune.
    
    # For a given S and candidate v, v can be added iff for all pairs (b,c) in S,
    # v is not in forbid[b][c] and v is not in forbid[c][b].
    # Also, for all b in S, v is not in forbid[v][b]... wait, the condition is:
    # for all distinct a,b,c in S∪{v}, a ∤ b*c.
    # If we add v to S, we need to check:
    # 1. For all distinct b,c in S: v ∤ b*c (i.e., v not in forbid[v][b] for the c... wait)
    #    Actually: v | b*c? No, we need v ∤ b*c. So check if v divides b*c.
    #    This is equivalent to: for the triple {v,b,c}, we need v ∤ b*c, b ∤ v*c, c ∤ v*b.
    #    b ∤ v*c iff c not in forbid[b][v]... let me re-examine.
    #    
    #    forbid[a][b] contains c such that a|b*c.
    #    So for triple {v,b,c}:
    #    - v|b*c iff c in forbid[v][b]
    #    - b|v*c iff c in forbid[b][v]  
    #    - c|v*b iff v in forbid[c][b]
    #    
    #    So v can be added to S iff for all distinct b,c in S:
    #    c not in forbid[v][b] AND c not in forbid[b][v] AND v not in forbid[c][b]
    
    # This check is O(|S|^2) per candidate, which might be slow.
    # Let's use a simpler but effective approach: greedy with local search, or
    # a smarter branch and bound.
    
    # Given the 60s limit, let's try a different approach:
    # Start with all primes > 128 (since for a > 128, b*c > 128*129 > a for most b,c... no)
    # Actually, if a > n/2, then b*c < a implies... no, b*c can be large.
    
    # Let's just do a simple greedy: iterate through vertices in some order, add if compatible.
    # Then try to improve.
    
    # Better: use the observation that if all numbers are primes, then a|b*c implies a|b or a|c (by Euclid's lemma),
    # so if a,b,c are distinct primes, a ∤ b*c. So all primes in 2..256 form a valid set.
    # Number of primes in 2..256: let's count.
    
    # Can we add composite numbers? If we add a composite a, we need that for all b,c in the set,
    # a ∤ b*c. If b and c are primes not dividing a, this is fine. But if b or c shares factors with a,
    # it might fail.
    
    # Let's start with all primes and see if we can add any composites.
    
    primes = [a for a in range(2, n + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    # Try to add composites one by one
    S = set(primes)
    for a in range(2, n + 1):
        if a in S:
            continue
        # Check if a can be added
        can_add = True
        for b in S:
            for c in S:
                if b == c or b == a or c == a:
                    continue
                # Check triple {a, b, c}
                if a % (b * c) == 0 or b % (a * c) == 0 or c % (a * b) == 0:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            S.add(a)
    
    return sorted(S)

# EVOLVE-BLOCK-END
