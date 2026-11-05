"""Improved search for Schütte's problem (k=4): find smallest Paley tournament valid."""
import time
import math

# Hard deadline: evaluator kills at 150s; we want to return well before that.
_DEADLINE = 145.0  # seconds


def _is_paley_valid(p):
    """Check if Paley tournament on p vertices satisfies the k=4 property.
    
    Property: Every set of 4 vertices must have a vertex with arcs to all four.
    Equivalently: For every 4-subset {a,b,c,d}, there exists v in {a,b,c,d}
    such that v beats all other three in the subset.
    
    A Paley tournament on prime p: i -> j iff (j-i) mod p is a quadratic residue.
    """
    if p < 5:
        return False
    
    # Precompute quadratic residues
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # For vertex v, the set of vertices it beats:
    # v -> u iff (u - v) % p is in squares
    # So v beats u iff (u - v) % p in squares
    # v loses to u iff (u - v) % p not in squares (and u != v)
    
    # For the k=4 property: for every 4-set {a,b,c,d}, at least one vertex
    # in the set beats the other three.
    #
    # Equivalently, there is NO 4-set where every vertex loses to at least one
    # other vertex in the set.
    #
    # A vertex v in a 4-set {v, x, y, z} does NOT dominate the set if v loses
    # to at least one of {x, y, z}.
    #
    # So the property fails iff there exists a 4-set where every vertex loses
    # to at least one other in the set.
    
    # Alternative characterization: In a tournament, a 4-set has no dominator
    # iff it's not a transitive 4-tournament and not a 4-cycle with a source...
    # Actually let's just check directly but efficiently.
    
    # For each vertex v, let L(v) = set of vertices that v loses to (i.e., u -> v)
    # v loses to u iff (v - u) % p in squares, i.e., u -> v iff (v-u)%p in squares
    # Wait: u -> v iff (v - u) % p is a square.
    # So v loses to u iff (v - u) % p in squares.
    # v beats u iff (u - v) % p in squares.
    
    # For a 4-set S = {a,b,c,d}, a vertex a dominates S if a beats b, c, d.
    # a beats b iff (b-a)%p in squares.
    
    # The property holds iff for every 4-set, at least one vertex beats all others.
    # This is equivalent to: there is no 4-set that is a "cycle" in the sense
    # that no single vertex dominates all three others.
    
    # Efficient check: For each vertex v, the 4-sets where v is the unique
    # dominator are those where the other 3 are all in N^+(v) (out-neighbors).
    # But we need at least ONE dominator per 4-set.
    
    # Let's think about when a 4-set has NO dominator:
    # A 4-set {a,b,c,d} has no dominator iff for each vertex in the set,
    # it loses to at least one other vertex in the set.
    
    # This is equivalent to the induced sub-tournament on the 4-set having
    # no source vertex.
    
    # For a Paley tournament, we can use the structure. But for small p,
    # a direct check is feasible.
    
    # For p up to ~100, C(p,4) can be up to ~3.9M for p=100.
    # For each 4-set, checking if any vertex dominates takes O(4*3) = 12 checks.
    # Total: ~47M operations. That's tight in Python.
    
    # Better approach: For each vertex v, let out(v) = set of vertices v beats.
    # A 4-set {v, x, y, z} has v as a dominator iff {x,y,z} ⊆ out(v).
    # So for each v, the number of 4-sets where v is a dominator is C(|out(v)|, 3).
    # But we need to check if EVERY 4-set has at least one dominator.
    
    # Alternative: A 4-set has no dominator iff it is NOT contained in out(v) ∪ {v}
    # for any v in the set... this is getting complex.
    
    # Let's use a different characterization:
    # In a tournament, a 4-set has a dominator (source) iff it is not one of
    # the "irreducible" 4-tournaments. The 4-vertex tournaments are:
    # 1. Transitive (has a source and sink) - HAS a dominator
    # 2. Cycle with one vertex dominating two (a 3-cycle plus one source) - HAS a dominator
    # 3. Two disjoint 2-cycles... no, tournaments don't have 2-cycles.
    # 4. The "regular" 4-tournament doesn't exist (needs odd number for regular).
    #
    # Actually for 4 vertices, the possible score sequences are:
    # (3,2,1,0) - transitive, has dominator
    # (3,1,1,1) - one vertex beats all, has dominator
    # (2,2,2,0) - no wait, sum must be 6. (2,2,1,1)? Let me recount.
    # Sum of scores = C(4,2) = 6.
    # Possible: (3,2,1,0), (3,1,1,1), (2,2,2,0) - no, (2,2,2,0) sums to 6 but
    # can a vertex have score 2 and another also 2? Yes.
    # (2,2,1,1) sums to 6.
    #
    # Which of these have a source (vertex with score 3)?
    # (3,2,1,0): yes, the vertex with score 3.
    # (3,1,1,1): yes, the vertex with score 3.
    # (2,2,2,0): no vertex has score 3, so NO dominator.
    # (2,2,1,1): no vertex has score 3, so NO dominator.
    
    # So we need: no 4-set induces a tournament with score sequence (2,2,2,0) or (2,2,1,1).
    
    # For a 4-set {a,b,c,d}, the scores within the set determine if there's a dominator.
    # Vertex a has score 3 in the set iff a beats b, c, d.
    
    # Let's just do a direct check but optimized:
    # Precompute for each pair (i,j) with i<j: does i beat j?
    # beat[i][j] = 1 if i->j, 0 otherwise (for i<j)
    
    # Actually, let me just check all 4-sets for small p.
    # For p <= 40, C(p,4) <= 91390, very fast.
    # For p = 50, C(50,4) = 230300, still fast.
    # For p = 60, C(60,4) = 487635, okay.
    # For p = 70, C(70,4) = 916895, okay.
    # For p = 80, C(80,4) = 1581580, might be slow in pure Python.
    # For p = 83, C(83,4) = 1916747, slow.
    
    # Let's use a smarter check. For each vertex v, compute out(v).
    # A 4-set has a dominator iff it's not the case that all four vertices
    # have out-degree <= 2 within the set.
    
    # Actually, the cleanest check: a 4-set {a,b,c,d} has NO dominator iff
    # for each vertex in the set, it has at least one out-neighbor AND at least
    # one in-neighbor within the set. I.e., no vertex has out-degree 3 or 0 in the set.
    # Wait, that's not right either. A vertex with out-degree 0 in the set (score 0)
    # doesn't help, but the set could still have a dominator (another vertex with score 3).
    
    # Let me just check: does any vertex in the 4-set beat all other three?
    
    # For efficiency with p up to ~100, let me use numpy.
    
    import numpy as np
    
    # Build adjacency matrix: A[i][j] = 1 if i -> j
    A = np.zeros((p, p), dtype=np.uint8)
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                A[i][j] = 1
                A[j][i] = 0
            else:
                A[i][j] = 0
                A[j][i] = 1
    
    # For each 4-set, check if any vertex dominates.
    # Use the fact that a 4-set has no dominator iff it's a "regular-like" or 
    # specific non-transitive pattern.
    
    # Let's enumerate 4-sets but with early termination.
    # For speed, iterate over 4-sets and check.
    
    # Actually, a key insight: in a Paley tournament, the structure is highly symmetric.
    # Let's check if there exists a 4-set with no dominator.
    
    # For each vertex a, consider 4-sets containing a: {a, b, c, d}.
    # a dominates the set iff b, c, d are all out-neighbors of a.
    # If a does NOT dominate, then at least one of b,c,d is an in-neighbor of a.
    
    # The set has NO dominator iff:
    # - a doesn't dominate (at least one of b,c,d beats a)
    # - b doesn't dominate (at least one of a,c,d beats b)  
    # - c doesn't dominate
    # - d doesn't dominate
    
    # This is complex to check vectorially for all 4-sets.
    
    # Alternative: just iterate. For p up to 67, C(67,4) = 736281.
    # For each 4-set, 4 checks of 3 comparisons = 12 ops. Total ~8.8M ops.
    # In Python this might take ~5-10 seconds. Acceptable.
    
    # Let me do it with early exit.
    for a in range(p):
        for b in range(a + 1, p):
            for c in range(b + 1, p):
                for d in range(c + 1, p):
                    # Check if any of a,b,c,d dominates the other three
                    # a dominates iff a->b, a->c, a->d
                    if A[a][b] and A[a][c] and A[a][d]:
                        continue  # a is dominator, this set is fine
                    # b dominates iff b->a, b->c, b->d
                    if A[b][a] and A[b][c] and A[b][d]:
                        continue
                    # c dominates iff c->a, c->b, c->d
                    if A[c][a] and A[c][b] and A[c][d]:
                        continue
                    # d dominates iff d->a, d->b, d->c
                    if A[d][a] and A[d][b] and A[d][c]:
                        continue
                    # No dominator found
                    return False
    return True


def construct():
    start_time = time.time()
    
    # Search over primes from small to large, find the smallest valid Paley tournament.
    # We know from the knowledge that checking primes 31-71 found an improvement (score 1.239).
    # The seed used p=83. Let's search from 31 upward.
    
    # Primes in range [31, 90]
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [p for p in range(31, 91) if is_prime(p)]
    
    best_n = None
    best_orientation = None
    
    for p in primes:
        # Check deadline
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        if _is_paley_valid(p):
            best_n = p
            # Build orientation
            squares = {x * x % p for x in range(1, p)}
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    orientation.append(0 if (j - i) % p in squares else 1)
            best_orientation = orientation
            break  # Found the smallest valid one
    
    # If we found one, return it
    if best_n is not None:
        return {'n': best_n, 'orientation': best_orientation}
    
    # Fallback: return the original Paley tournament on 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
