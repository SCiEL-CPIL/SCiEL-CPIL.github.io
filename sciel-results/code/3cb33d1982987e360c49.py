"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    V = 16
    all_triples = list(combinations(range(V), 3))
    best_triples = []
    
    # Initial seed from current program strategy for safety
    # But we will try to improve it via greedy search
    # The current program returns all triples with 0: C(15,2) = 105 triples.
    # We can likely do better by mixing points.
    
    # To ensure we always have a valid baseline, we keep the best found.
    # We will run multiple random greedy attempts.
    
    def is_k222_free(triples_set, new_triple):
        """
        Check if adding new_triple to triples_set creates a K(2,2,2).
        A K(2,2,2) consists of 3 disjoint pairs (a,a'), (b,b'), (c,c')
        such that all 8 transversal triples are present.
        
        If we add (x,y,z), the only way it creates a K(2,2,2) is if there exist
        x', y', z' such that:
        1. {x,x'}, {y,y'}, {z,z'} are disjoint pairs (i.e., x!=x', y!=y', z!=z', and all 6 distinct)
        2. The other 7 transversal triples are already in triples_set.
        
        The 8 triples are:
        (x,y,z) - the new one
        (x',y,z)
        (x,y',z)
        (x,y,z')
        (x',y',z)
        (x',y,z')
        (x,y',z')
        (x',y',z')
        
        So we need to check if there exist x', y', z' distinct from x,y,z and each other
        such that the 7 other triples are in triples_set.
        """
        x, y, z = new_triple
        # Candidates for x' must be distinct from x,y,z
        # Candidates for y' must be distinct from x,y,z and x'
        # Candidates for z' must be distinct from x,y,z, x', y'
        
        # To optimize, we can iterate over potential x', y', z'.
        # There are 13 candidates for x' (excluding x,y,z).
        # For each x', 12 candidates for y'.
        # For each (x', y'), 11 candidates for z'.
        # Total checks: 13 * 12 * 11 = 1716.
        # Each check involves 7 set lookups.
        # This is fast enough for a greedy step if we don't do it for every single triple in a huge list, 
        # but we are iterating through ~560 triples. 560 * 1716 * 7 ~ 6.7 million operations.
        # Python might be slow, but with 100s budget, we can afford it.
        
        # Optimization: Pre-compute candidates?
        # Let's just iterate.
        
        candidates = [v for v in range(V) if v not in (x, y, z)]
        
        for xp in candidates:
            for yp in candidates:
                if yp == xp: continue
                for zp in candidates:
                    if zp == xp or zp == yp: continue
                    
                    # Check the 7 other triples
                    # (xp, y, z)
                    if (xp, y, z) not in triples_set: continue
                    # (x, yp, z)
                    if (x, yp, z) not in triples_set: continue
                    # (x, y, zp)
                    if (x, y, zp) not in triples_set: continue
                    # (xp, yp, z)
                    if (xp, yp, z) not in triples_set: continue
                    # (xp, y, zp)
                    if (xp, y, zp) not in triples_set: continue
                    # (x, yp, zp)
                    if (x, yp, zp) not in triples_set: continue
                    # (xp, yp, zp)
                    if (xp, yp, zp) not in triples_set: continue
                    
                    # If we reach here, all 7 exist, so adding (x,y,z) completes the K(2,2,2)
                    return False
        return True

    def greedy_randomized(deadline):
        triples_set = set()
        triples_list = []
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        for t in shuffled:
            if time.time() > deadline:
                break
            if is_k222_free(triples_set, t):
                triples_set.add(t)
                triples_list.append(list(t))
        
        return triples_list

    # Run multiple random restarts
    # Each restart takes time. We run as many as we can.
    while time.time() < deadline:
        current = greedy_randomized(deadline)
        if len(current) > len(best_triples):
            best_triples = current
            
        # If we hit the deadline inside greedy, we break out
        if time.time() > deadline:
            break

    # Fallback to seed if no better found (shouldn't happen as seed is valid)
    if not best_triples:
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
        
    return best_triples
# EVOLVE-BLOCK-END
