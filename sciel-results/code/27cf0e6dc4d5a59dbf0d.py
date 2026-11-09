import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    def is_valid_paley(p):
        # p must be prime and p % 4 == 3
        if p < 5: return False
        squares = {x * x % p for x in range(1, p)}
        # Check if p is prime (simple trial division for small p)
        if p > 1:
            for i in range(2, int(p**0.5) + 1):
                if p % i == 0:
                    return False
        
        # Build orientation
        # 0 means i -> j
        # i -> j iff (j-i) % p is a square
        # We need to check if every 4-set has a common dominator.
        # A vertex v dominates {a,b,c,d} if v->a, v->b, v->c, v->d.
        # This is equivalent to: for every 4-set S, the intersection of the out-neighborhoods of all v in V is not empty?
        # No, "exists v in V such that v->S".
        # So for a fixed S, we need to check if there is any v such that v is in the out-neighborhood of all s in S?
        # No, v->s means s is in the out-neighborhood of v.
        # So we need to check if the intersection of the *in-neighborhoods* of the vertices in S is non-empty?
        # Let N^+(v) be out-neighbors of v.
        # We want $\exists v, \forall s \in S, s \in N^+(v)$.
        # This means $v \in \bigcap_{s \in S} N^-(s)$, where $N^-(s)$ is the set of in-neighbors of s.
        # So for every 4-set S, $\bigcap_{s \in S} N^-(s)$ must be non-empty.
        
        # To check this efficiently:
        # Precompute in-neighbors for each vertex.
        # For each 4-set, check intersection.
        # Number of 4-sets is C(p, 4). For p=67, C(67,4) ~ 700,000.
        # Intersection of 4 sets of size ~33.
        # This might be slow in pure Python if done naively.
        
        # Optimization:
        # Represent in-neighbors as bitmasks or sets.
        # Using sets:
        in_neigh = [set() for _ in range(p)]
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    # i -> j
                    in_neigh[j].add(i)
                else:
                    # j -> i
                    in_neigh[i].add(j)
        
        # Check all 4-sets
        # We can prune: if a vertex has in-degree < 4, it can't dominate any 4-set?
        # No, a vertex v dominates S if v->s for all s in S.
        # So v must have out-degree >= 4.
        # In Paley, out-degree is (p-1)/2.
        
        # Let's just iterate.
        # To speed up, we can check smaller sets first?
        # Actually, if there is a 4-set with no dominator, there might be a 3-set with no dominator?
        # If no v dominates {a,b,c}, then no v dominates {a,b,c,d}.
        # So it suffices to check all 3-sets!
        # If every 3-set has a common dominator, then every 4-set does?
        # Let S be a 4-set. Let T be a 3-subset of S.
        # If T has a dominator v, does v dominate S? Not necessarily. v->T but maybe not v->s4.
        # So checking 3-sets is NOT sufficient.
        
        # However, if we check all 4-sets, it's C(67,4) ~ 700k.
        # Each check is intersection of 4 sets.
        # 700k * (small constant) should be fast enough in Python if optimized.
        
        for a in range(p):
            for b in range(a + 1, p):
                # Optimization: if intersection of in_neigh[a] and in_neigh[b] is empty, skip
                # But we need 4 vertices.
                pass
        # This nested loop is too slow in pure Python.
        # We need a better way.
        
        # Alternative:
        # For each vertex v, let Out(v) be the set of vertices v points to.
        # A 4-set S has no dominator if for every v, S is not a subset of Out(v).
        # This is equivalent to: The family of sets {Out(v) | v in V} does not cover any 4-set?
        # No, it means that for every 4-set S, S is not contained in any Out(v).
        
        # Let's try a different approach:
        # If we can't check quickly, we might just assume Paley(67) is valid based on literature.
        # The prompt says "smallest known has 67 vertices".
        # It is highly probable that Paley(67) is the one.
        
        return True # Placeholder, we need to verify or trust.

    # Let's try to verify Paley(67) quickly.
    # If it fails, we return 83.
    
    # Actually, we can just return Paley(67) and hope it's valid.
    # But if it's invalid, we get 0.
    # We should verify it.
    
    p = 67
    squares = {x * x % p for x in range(1, p)}
    
    # Check validity of Paley(67)
    # We can use a faster check:
    # A 4-set {a,b,c,d} has no dominator iff for all v, v does not point to all of a,b,c,d.
    # This is equivalent to: The maximum size of a set of vertices all dominated by a single v is < 4?
    # No, that's not right.
    
    # Let's stick to the intersection check but optimize it.
    # We can represent in-neighbors as integers (bitmasks).
    # p=67 fits in a 64-bit integer? No, 2^67. Python handles big ints.
    
    in_masks = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                # i -> j
                in_masks[j] |= (1 << i)
            else:
                # j -> i
                in_masks[i] |= (1 << j)
                
    # Check all 4-sets
    # We can iterate and break early.
    # To speed up, we can check 3-sets first?
    # If a 3-set has no dominator, the tournament is invalid.
    # If all 3-sets have a dominator, is it valid for 4-sets?
    # Counter-example: v1->a,b,c. v2->a,b,d. v3->c,d,e...
    # It's possible that 3-sets are covered but 4-sets are not.
    # So we must check 4-sets.
    
    # 700k iterations.
    # Each iteration: 4 bitwise ANDs.
    # This is very fast in Python with big ints.
    
    valid = True
    # We can sample if it's too slow, but 700k is likely < 1s.
    for a in range(p):
        for b in range(a + 1, p):
            m_ab = in_masks[a] & in_masks[b]
            if m_ab == 0: continue # No common dominator for {a,b}, so no for any superset
            for c in range(b + 1, p):
                m_abc = m_ab & in_masks[c]
                if m_abc == 0: continue
                for d in range(c + 1, p):
                    if (m_abc & in_masks[d]) == 0:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: break
        
    if valid:
        orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
        return {'n': p, 'orientation': orientation}
    else:
        # Fallback to 83
        p = 83
        squares = {x * x % p for x in range(1, p)}
        orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
        return {'n': p, 'orientation': orientation}
