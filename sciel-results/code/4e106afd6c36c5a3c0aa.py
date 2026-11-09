import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    deadline = time.time() + 140
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_set = set(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # For each triple, find all K(2,2,2)s it belongs to
    # A K(2,2,2) is (a1,a2),(b1,b2),(c1,c2) with 6 distinct points
    # Triple {x,y,z} is in K(2,2,2) if x,y,z are in different pairs
    # For a given triple {x,y,z}, the K(2,2,2)s containing it:
    #   x is paired with some x', y with y', z with z'
    #   x' != y', z', x' != y, z, etc. (all 6 distinct)
    # So for triple {x,y,z}: choose x' from remaining 13 points (not x,y,z),
    #   then y' from remaining 12, then z' from remaining 11
    # But we need to count each K(2,2,2) once. The K(2,2,2) is determined by
    # the three pairs, so {x,x'}, {y,y'}, {z,z'} defines it uniquely given the triple.
    # Number of K(2,2,2)s per triple: 13 * 12 * 11 = 1716? No wait...
    # Actually x' can be any of the 13 points not in {x,y,z}. Then y' any of 12 not in {x,y,z,x'}. Then z' any of 11.
    # But this overcounts because the K(2,2,2) is the same regardless of which triple we look at it from.
    # For our purposes, we just need: for each triple, which K(2,2,2)s does it belong to?
    # And for each K(2,2,2), which 8 triples does it contain?
    
    # Let's enumerate all K(2,2,2)s
    # A K(2,2,2) is 6 distinct points partitioned into 3 unordered pairs
    # Number: C(16,6) * (number of ways to partition 6 into 3 pairs) = 8008 * 15 = 120120
    
    # For each K(2,2,2), the 8 transversal triples
    # Let's precompute: for each triple, the list of K(2,2,2) indices it belongs to
    # And for each K(2,2,2), the list of 8 triple indices
    
    # This might be memory heavy. Let's be smart.
    # Alternative: for each triple, compute which other triples form a K(2,2,2) completion with it.
    # Actually, let's think differently.
    
    # For greedy: we want to check if adding triple T creates a K(2,2,2) with all 8 present.
    # T belongs to some K(2,2,2)s. For each such K(2,2,2), if the other 7 triples are all present, we can't add T.
    
    # Let's precompute for each triple the list of K(2,2,2) IDs it's in.
    # But 120120 K(2,2,2)s * 8 triples each = 960960 triple-K(2,2,2) incidences
    # And 560 triples * 1716 K(2,2,2)s each = 960960. Same number. Good.
    
    # Let's enumerate K(2,2,2)s
    k222_triples = []  # list of (8 triple tuples)
    triple_to_k222 = defaultdict(list)  # triple -> list of k222 indices
    
    # Enumerate all 6-subsets and their pairings
    # For 6 points p0..p5, the 15 pairings:
    # Actually, let's just iterate over all ways to pick 3 disjoint pairs from 16 points
    # This is expensive. Let me think of a better way.
    
    # Alternative approach: for each triple T={x,y,z}, the K(2,2,2)s it's in are determined by
    # choosing partners x', y', z' from the other 13 points.
    # For greedy checking, I don't need to enumerate all K(2,2,2)s upfront.
    # Instead, for each triple T, I can check: does adding T complete any K(2,2,2)?
    # T is in a K(2,2,2) with partners x', y', z' iff {x,x'}, {y,y'}, {z,z'} are the 3 pairs.
    # The 8 transversal triples are {a,b,c} where a∈{x,x'}, b∈{y,y'}, c∈{z,z'}.
    # So the other 7 are all combinations except {x,y,z} itself.
    
    # For greedy: maintain a set of selected triples.
    # To check if triple T can be added:
    #   For each partner triple (x',y',z') from remaining 13 points:
    #     Check if the other 7 transversal triples are all selected.
    #   If any such completion exists, T cannot be added.
    
    # This check is O(13*12*11) = 1716 per triple, each check is 7 set lookups.
    # For 560 triples, that's about 560*1716*7 ≈ 6.7 million operations per full scan.
    # With multiple restarts, this should be feasible.
    
    def can_add(T, selected_set):
        x, y, z = T
        # x', y', z' must be distinct and not in {x,y,z}
        others = [p for p in range(n) if p not in T]
        for xp in others:
            for yp in others:
                if yp == xp: continue
                for zp in others:
                    if zp == xp or zp == yp: continue
                    # Check other 7 transversal triples
                    # Pairs: (x,xp), (y,yp), (z,zp)
                    # 8 triples: all {a,b,c} where a in (x,xp), b in (y,yp), c in (z,zp)
                    # T = {x,y,z}, so other 7:
                    # {xp,y,z}, {x,yp,z}, {x,y,zp}, {xp,yp,z}, {xp,y,zp}, {x,yp,zp}, {xp,yp,zp}
                    if (tuple(sorted((xp,y,z))) in selected_set and
                        tuple(sorted((x,yp,z))) in selected_set and
                        tuple(sorted((x,y,zp))) in selected_set and
                        tuple(sorted((xp,yp,z))) in selected_set and
                        tuple(sorted((xp,y,zp))) in selected_set and
                        tuple(sorted((x,yp,zp))) in selected_set and
                        tuple(sorted((xp,yp,zp))) in selected_set):
                        return False
        return True
    
    best_triples = []
    
    # Strategy: multiple random-order greedy passes, keep best
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        order = list(all_triples)
        random.shuffle(order)
        
        selected = set()
        selected_list = []
        
        for T in order:
            if time.time() > deadline - 5:
                break
            if can_add(T, selected):
                selected.add(T)
                selected_list.append(list(T))
        
        if len(selected_list) > len(best_triples):
            best_triples = selected_list[:]
    
    # Local search: try to improve by swapping
    # Try removing one triple and adding two that were previously blocked
    if time.time() < deadline - 15:
        selected = set(tuple(t) for t in best_triples)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            # Try adding any unselected triple
            for T in all_triples:
                if time.time() > deadline - 15:
                    break
                if T not in selected and can_add(T, selected):
                    selected.add(T)
                    best_triples = [list(t) for t in selected]
                    improved = True
                    break
    
    return best_triples
