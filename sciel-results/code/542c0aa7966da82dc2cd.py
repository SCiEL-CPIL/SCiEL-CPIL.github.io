"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # We aim for n=67, which is the smallest known valid size for this property.
    # The Paley tournament on p=67 (p=4k+3 prime) is a strong candidate.
    # Seed used p=83.
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Keep 10s safety margin
    
    def is_valid_paley(p):
        """
        Checks if the Paley tournament on p vertices satisfies:
        Every set of 4 vertices has a common dominator.
        i -> j iff (j-i) % p is a quadratic residue mod p (nonzero).
        """
        if p < 4:
            return True
        
        squares = {x * x % p for x in range(1, p)}
        
        # Precompute out-neighborhoods as bitmasks for speed?
        # n=67 is small enough for sets, but bitmasks are faster for intersection checks.
        # However, checking "S subset of N+(v)" for all S of size 4 is O(n^4 * n) which is ~67^5 ~ 1.3e9. Too slow in Python.
        
        # Alternative check:
        # A 4-set S does NOT have a common dominator if the intersection of the "in-neighborhoods" of all u in S is empty?
        # No, we need v such that v->u for all u in S.
        # This means v is in the intersection of the "in-neighborhoods" of u? No.
        # v->u means u is in N+(v).
        # So v must be in the intersection of { w : w->u } for all u in S.
        # Let P(u) be the set of vertices that point TO u (in-neighborhood).
        # We need intersection_{u in S} P(u) to be non-empty.
        
        # So the condition is: For all S of size 4, intersection_{u in S} P(u) is not empty.
        
        # How to check this efficiently?
        # Iterate through all 4-sets? \binom{67}{4} \approx 190,000.
        # For each 4-set, compute intersection of 4 bitmasks.
        # 190,000 * 4 operations on Python ints (bitmasks) is very fast.
        
        # Build in-neighborhood bitmasks
        # in_nb[u] is a bitmask where bit v is 1 if v -> u.
        in_nb = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    # i -> j
                    # j is in N+(i), so i is in P(j)
                    in_nb[j] |= (1 << i)
                else:
                    # j -> i
                    # i is in N+(j), so j is in P(i)
                    in_nb[i] |= (1 << j)
        
        # Check all 4-sets
        # To avoid generating all 4-sets explicitly if possible, or just do it.
        # 190k iterations is fine.
        for a in range(p - 3):
            for b in range(a + 1, p - 2):
                # Pre-intersect a and b
                mask_ab = in_nb[a] & in_nb[b]
                if mask_ab == 0:
                    # If intersection of first two is empty, any superset will be empty.
                    # But we need to check if there's ANY 4-set with empty intersection.
                    # If mask_ab is 0, then for any c, d, intersection is 0.
                    # So if we find a pair (a,b) with empty in-neighborhood intersection, 
                    # does it imply a 4-set failure?
                    # We need a 4-set {a,b,c,d}. If P(a) & P(b) is empty, then no vertex points to BOTH a and b.
                    # Thus no vertex points to a, b, c, d.
                    # So if there exists ANY pair {a,b} such that P(a) & P(b) is empty, 
                    # then the property fails?
                    # Yes, because we can pick any c, d distinct from a, b.
                    # The intersection P(a)&P(b)&P(c)&P(d) is a subset of P(a)&P(b), which is empty.
                    # So the condition "Every 4-set has a dominator" implies 
                    # "Every 2-set has a common dominator".
                    # Is this true?
                    # If every 4-set has a dominator, then specifically for any 4-set containing a and b, 
                    # there is a dominator. That dominator must point to a and b.
                    # So yes, if the property holds for k=4, it must hold for k=2.
                    # So we just need to check if there is any pair with empty common in-neighborhood.
                    
                    return False
                
                for c in range(b + 1, p - 1):
                    mask_abc = mask_ab & in_nb[c]
                    if mask_abc == 0:
                        # Pick any d > c
                        # Intersection will be empty.
                        return False
                    
                    # We don't even need to loop d, because if P(a)&P(b)&P(c) is empty,
                    # then for any d, the intersection is empty.
                    # So the condition is equivalent to: 
                    # Every 3-set has a common dominator.
                    # And every 2-set has a common dominator.
                    # Actually, if every 3-set has a common dominator, does every 4-set?
                    # Let S = {a,b,c,d}.
                    # We know there is v_ab in P(a)&P(b).
                    # We know there is v_ac in P(a)&P(c).
                    # This doesn't guarantee a single v in all 4.
                    # So we must check 4-sets?
                    # Wait, my logic above:
                    # If P(a)&P(b) is empty, then NO 4-set containing a,b has a dominator.
                    # So checking pairs is a necessary condition.
                    # If P(a)&P(b)&P(c) is empty, then NO 4-set containing a,b,c has a dominator.
                    # So checking triples is a necessary condition.
                    # If P(a)&P(b)&P(c)&P(d) is empty, then that specific 4-set fails.
                    
                    # So I need to check if there exists ANY 4-set with empty intersection.
                    # The loop above returns False if it finds a 3-set with empty intersection.
                    # Is that sufficient?
                    # If P(a)&P(b)&P(c) is empty, then for ANY d, P(a)&P(b)&P(c)&P(d) is empty.
                    # So yes, if any 3-set has empty intersection, the property fails.
                    # Conversely, if all 3-sets have non-empty intersection, do all 4-sets?
                    # No. It is possible that P(a)&P(b)&P(c) is not empty, and P(a)&P(b)&P(d) is not empty,
                    # but their intersection (with the third one) is empty?
                    # Let X = P(a)&P(b).
                    # We need X & P(c) & P(d) != empty.
                    # We know X & P(c) != empty (from triple check).
                    # Let Y = X & P(c). Y is non-empty.
                    # We need Y & P(d) != empty.
                    # The triple check for {a,b,d} ensures X & P(d) != empty.
                    # But it does NOT ensure (X & P(c)) & P(d) != empty.
                    # Example: X={1,2}, P(c)={1}, P(d)={2}.
                    # X&P(c)={1} (non-empty).
                    # X&P(d)={2} (non-empty).
                    # X&P(c)&P(d) = {1}&{2} = empty.
                    # So checking triples is NOT sufficient. I must check 4-sets.
                    
                    for d in range(c + 1, p):
                        if (mask_abc & in_nb[d]) == 0:
                            return False
        
        return True

    # Try p=67 first
    p = 67
    if is_valid_paley(p):
        squares = {x * x % p for x in range(1, p)}
        orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
        return {'n': p, 'orientation': orientation}
    
    # Fallback to p=83 if 67 fails (should not happen based on prompt)
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-END
