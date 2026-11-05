"""Paley tournament search: find smallest prime p ≡ 3 (mod 4) whose Paley tournament is 4-dominating."""
# EVOLVE-BLOCK-START
import time
import math

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _paley_is_4_dominating(p):
    """Check if Paley tournament on p vertices is 4-dominating.
    i -> j iff (j-i) mod p is a nonzero quadratic residue mod p.
    """
    if p < 5:
        return False
    # Build the set of quadratic residues (nonzero squares) mod p
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    # For the Paley tournament on p ≡ 3 (mod 4), the non-residues are the negatives of residues
    # Actually, let's just build the adjacency: i -> j iff (j-i) mod p in residues
    # For each vertex v, the set of vertices it dominates (out-neighbors) is:
    #   {v + r mod p : r in residues}
    # The set of vertices that dominate v (in-neighbors) is:
    #   {v - r mod p : r in residues} = {v + s mod p : s in non-residues}
    #
    # We need: for every 4-subset S, there exists v in S such that v dominates all other 3 in S.
    # Equivalently: for every 4-subset S, there is v in S with S \ {v} ⊆ Out(v).
    #
    # A faster approach: the tournament is NOT 4-dominating iff there exists a 4-subset S
    # such that for every v in S, there exists u in S\{v} with u not in Out(v),
    # i.e., for every v in S, there is u in S\{v} such that v does NOT dominate u,
    # i.e., (u - v) mod p is not in residues (i.e., it's a non-residue or 0, but u≠v so non-residue).
    #
    # So S is "bad" iff for every v in S, there exists u in S\{v} with (u-v) mod p ∈ non-residues.
    #
    # Let's think about this differently. For a fixed 4-set {a,b,c,d}, it's bad iff:
    #   for each of the 4 vertices, at least one of the other 3 is an in-neighbor.
    #
    # We can check: for each 4-set, compute for each vertex how many of the other 3 it dominates.
    # If any vertex dominates all 3, it's good. Otherwise it's bad.
    #
    # But checking all C(p,4) sets is too slow for large p.
    #
    # Alternative: The Paley tournament is a strongly regular graph structure.
    # For p ≡ 3 (mod 4), the Paley tournament is a (p, (p-1)/2, (p-5)/4, (p-1)/4) design... 
    # Actually it's a tournament, so let's think in terms of the difference set.
    #
    # The key insight: In a Paley tournament, the out-neighborhood of any vertex v is 
    # v + R where R is the set of quadratic residues. The structure is vertex-transitive.
    #
    # A 4-set {a,b,c,d} is bad iff for each element, it has at least one in-neighbor in the set.
    # By translation invariance, we can assume a=0 and check sets {0, b, c, d} with 0 < b < c < d < p.
    # But that's still C(p-1,3) sets.
    #
    # Let me think of a smarter check. 
    # For the set {0, b, c, d}:
    #   0 is dominated by b iff b is a non-residue (since 0->b iff b is residue, so b->0 iff b is non-residue)
    #   Wait: 0 -> b iff (b-0) mod p = b is in residues. So 0 dominates b iff b ∈ R.
    #   b -> 0 iff (0-b) mod p = p-b is in R. Since p ≡ 3 (mod 4), -1 is a non-residue,
    #   so p-b ∈ R iff -(b) ∈ R iff b ∉ R (since -1 non-residue means x∈R iff -x∉R).
    #   So b -> 0 iff b ∉ R.
    #
    # Similarly, for any pair (x,y): x -> y iff (y-x) mod p ∈ R.
    #
    # So for {0, b, c, d}:
    #   0 dominates b iff b ∈ R; 0 dominates c iff c ∈ R; 0 dominates d iff d ∈ R.
    #   b dominates 0 iff b ∉ R; b dominates c iff (c-b) mod p ∈ R; b dominates d iff (d-b) mod p ∈ R.
    #   etc.
    #
    # The set is good iff some vertex dominates all other 3.
    #
    # For efficiency, let's use numpy and vectorize over the 4-sets... but that's memory heavy.
    #
    # Actually, let me just try small primes first and see which work.
    # The smallest known is 67. Let me check primes 3, 7, 11, 19, 23, 31, 43, 47, 59, 67, ...
    
    # Build adjacency as a boolean matrix for speed
    # adj[i][j] = True iff i -> j
    # adj[i][j] = (j - i) % p in residues
    
    # Use a set for O(1) lookup
    rset = residues
    
    # For each 4-subset, check if it's dominated.
    # To speed up: precompute for each pair (i,j) whether i->j.
    # Then for a 4-set {a,b,c,d}, check:
    #   a dominates b,c,d: adj[a][b] and adj[a][c] and adj[a][d]
    #   b dominates a,c,d: adj[b][a] and adj[b][c] and adj[b][d]
    #   etc.
    
    # Let's use a 2D array for adj
    import numpy as np
    adj = np.zeros((p, p), dtype=bool)
    for i in range(p):
        for r in rset:
            adj[i, (i + r) % p] = True
    
    # Now check all 4-subsets. For p up to ~100, C(p,4) can be large but manageable.
    # For p=67, C(67,4) = 751,506. That's fine.
    
    # Vectorized approach: for each combination of 4, check.
    # Let's do it with itertools for clarity, and numpy for the adj lookups.
    
    from itertools import combinations
    
    for quad in combinations(range(p), 4):
        a, b, c, d = quad
        # Check if any vertex dominates the other three
        if adj[a, b] and adj[a, c] and adj[a, d]:
            continue
        if adj[b, a] and adj[b, c] and adj[b, d]:
            continue
        if adj[c, a] and adj[c, b] and adj[c, d]:
            continue
        if adj[d, a] and adj[d, b] and adj[d, c]:
            continue
        # Found a bad 4-set
        return False
    
    return True

def _find_paley_prime(deadline):
    """Find the smallest prime p ≡ 3 (mod 4) with p >= 5 whose Paley tournament is 4-dominating."""
    # Start from small primes and go up
    p = 3
    while True:
        if time.time() > deadline:
            return None
        if p >= 5 and p % 4 == 3 and _is_prime(p):
            if _paley_is_4_dominating(p):
                return p
        p += 2  # only check odd numbers
        if p > 100:
            # If we've gone past 100, something is wrong or it takes too long
            # The known answer is 67, so we should find it before this
            break
    
    return None

def _build_paley_orientation(p):
    """Build the orientation list for the Paley tournament on p vertices."""
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            # 0 means i -> j, 1 means j -> i
            # i -> j iff diff is in residues
            orientation.append(0 if diff in residues else 1)
    
    return orientation

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    # Try to find the smallest Paley prime that works
    best_p = None
    p = 3
    while p <= 100:
        if time.time() > deadline:
            break
        if p >= 5 and p % 4 == 3 and _is_prime(p):
            if _paley_is_4_dominating(p):
                best_p = p
                break
        p += 2
    
    if best_p is None:
        # Fallback: use 83 (the original) or try a few more
        # Let's just use 83 as fallback
        best_p = 83
    
    orientation = _build_paley_orientation(best_p)
    return {'n': best_p, 'orientation': orientation}
# EVOLVE-BLOCK-END
