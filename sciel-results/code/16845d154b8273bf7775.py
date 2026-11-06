import time
import random
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    V = 16
    all_triples = list(combinations(range(V), 3))
    
    # Precompute masks for each triple
    # mask is a 16-bit integer where bit i is 1 if i is in the triple
    triple_masks = [0] * len(all_triples)
    for i, t in enumerate(all_triples):
        m = 0
        for v in t:
            m |= (1 << v)
        triple_masks[i] = m
        
    # Precompute pair -> list of triples containing that pair
    # pair is (u, v) with u < v
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for j in range(3):
            for k in range(j+1, 3):
                p = (t[j], t[k]) if t[j] < t[k] else (t[k], t[j])
                if p not in pair_to_triples:
                    pair_to_triples[p] = []
                pair_to_triples[p].append(i)

    def is_valid_addition(trips_set, t_idx):
        """
        Check if adding triple t_idx to trips_set creates a K(2,2,2).
        A K(2,2,2) consists of 6 vertices partitioned into 3 pairs (A,B), (C,D), (E,F)
        such that all 8 transversal triples are present.
        The new triple t = {x,y,z} must be one of these 8.
        The other 7 must be in trips_set.
        
        Instead of checking all 8, we can look at the pairs formed by the vertices of t.
        Let t = {x,y,z}.
        If t is part of a K(2,2,2), then there exist vertices x', y', z' such that
        the set of 8 triples is fully present.
        The pairs in the K(2,2,2) are {x,x'}, {y,y'}, {z,z'}.
        The triples in the K(2,2,2) are all combinations of choosing one from each pair.
        Since t={x,y,z} is in the set, the other 7 are:
        {x',y,z}, {x,y',z}, {x,y,z'}, {x',y',z}, {x',y,z'}, {x,y',z'}, {x',y',z'}
        
        We iterate over possible x', y', z' disjoint from {x,y,z}.
        """
        t = all_triples[t_idx]
        x, y, z = t
        
        # Candidates for x' must be different from x, y, z
        # Candidates for y' must be different from x, y, z and x'
        # Candidates for z' must be different from x, y, z, x', y'
        
        # Optimization: 
        # For a K(2,2,2) to exist involving t, there must be a pair {x, x'} such that 
        # the triples {x, y, z} and {x', y, z} are present? No, {x',y,z} must be in trips_set.
        # Actually, if {x', y, z} is in trips_set, then the pair {y,z} is in both t and {x',y,z}.
        # Let's check pairs.
        
        # If we add t, we form new pairs (x,y), (x,z), (y,z).
        # Any K(2,2,2) using t must use at least one of these pairs as part of the partition?
        # No, the pairs in the K(2,2,2) partition are disjoint.
        # The vertices of t are x, y, z. They must belong to different pairs in the partition.
        # So the pairs in the partition are {x, xp}, {y, yp}, {z, zp}.
        
        # For this to be a K(2,2,2), all 8 triples must be in the graph.
        # The 8 triples are:
        # 1. {x, y, z} (t itself)
        # 2. {xp, y, z}
        # 3. {x, yp, z}
        # 4. {x, y, zp}
        # 5. {xp, yp, z}
        # 6. {xp, y, zp}
        # 7. {x, yp, zp}
        # 8. {xp, yp, zp}
        
        # We need 2..8 to be in trips_set.
        
        # We can iterate over xp, yp, zp.
        # To speed up, we can precompute for each pair (u,v) the set of triples containing it.
        # But checking 13*12*11 ~ 1700 combinations per addition is expensive if done naively in Python.
        
        # Better check:
        # For each pair (u,v) in { (x,y), (x,z), (y,z) }, this is not directly useful for the partition pairs.
        # The partition pairs are (x, xp), (y, yp), (z, zp).
        
        # Let's iterate over xp in range(V) \ {x,y,z}
        # Then we need {xp, y, z} in trips_set.
        # If so, then (xp, y, z) is a candidate.
        # Then we iterate yp in range(V) \ {x,y,z,xp}
        # Then we need {x, yp, z} in trips_set.
        # If so, then (x, yp, z) is a candidate.
        # Then we iterate zp in range(V) \ {x,y,z,xp,yp}
        # Then we need {x, y, zp} in trips_set.
        # If so, then we check the remaining 4:
        # {xp, yp, z}, {xp, y, zp}, {x, yp, zp}, {xp, yp, zp}
        
        # This reduces the search space significantly.
        
        for xp in range(V):
            if xp in (x, y, z): continue
            t2 = (min(xp, y), min(y, z), max(xp, y, z)) if False else None # Need proper tuple
            # Get index of {xp, y, z}
            # We can use a set of triples for O(1) lookup
            if (xp, y, z) not in trips_set and (xp, z, y) not in trips_set and (y, xp, z) not in trips_set and (y, z, xp) not in trips_set and (z, xp, y) not in trips_set and (z, y, xp) not in trips_set:
                # Faster: use a frozenset or sorted tuple in a set
                pass
            
        # Let's use a set of frozensets for trips_set for O(1) lookup.
        # But frozenset creation is slow.
        # Alternative: encode triple as integer.
        # idx = a*136 + b*16 + c for a<b<c? No, 16^3 = 4096.
        # Let's use a set of integers representing triples.
        
        # We'll pass a set of integers.
        
        for xp in range(V):
            if xp in (x, y, z): continue
            # Check if {xp, y, z} is in trips_set
            # Sort xp, y, z
            a, b, c = sorted((xp, y, z))
            if (a, b, c) not in trips_set: continue
            
            for yp in range(V):
                if yp in (x, y, z, xp): continue
                a, b, c = sorted((x, yp, z))
                if (a, b, c) not in trips_set: continue
                
                for zp in range(V):
                    if zp in (x, y, z, xp, yp): continue
                    a, b, c = sorted((x, y, zp))
                    if (a, b, c) not in trips_set: continue
                    
                    # Check remaining 4
                    a, b, c = sorted((xp, yp, z))
                    if (a, b, c) not in trips_set: continue
                    a, b, c = sorted((xp, y, zp))
                    if (a, b, c) not in trips_set: continue
                    a, b, c = sorted((x, yp, zp))
                    if (a, b, c) not in trips_set: continue
                    a, b, c = sorted((xp, yp, zp))
                    if (a, b, c) not in trips_set: continue
                    
                    return False # Invalid, creates K(2,2,2)
                    
        return True # Valid

    def greedy_construction(seed):
        random.seed(seed)
        trips_set = set()
        # Start with a random subset or empty?
        # Randomized greedy: shuffle all triples, try to add each one.
        # To get a good start, maybe add a random base set first?
        # Or just start empty and shuffle.
        
        # Strategy: Start with empty, shuffle triples, add if valid.
        # This is slow if we check every triple against a growing set.
        # Better: Start with a known valid configuration?
        # The seed program returns all triples through 0. That's 105 triples.
        # Let's try to improve upon that.
        
        # Let's use the seed program's output as a starting point for some runs,
        # and empty for others, to explore different landscapes.
        
        if seed % 2 == 0:
            # Start with star at 0
            for a in range(1, V):
                for b in range(a+1, V):
                    trips_set.add((0, a, b))
        else:
            # Start empty
            pass

        # Now try to add remaining triples
        candidates = [t for t in all_triples if t not in trips_set]
        random.shuffle(candidates)
        
        for t in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(trips_set, t):
                trips_set.add(t)
                
        return trips_set

    def local_search(trips_set, seed):
        random.seed(seed)
        current_set = set(trips_set)
        best_set = set(current_set)
        best_size = len(current_set)
        
        # Try to remove one triple and add two? Or just add more?
        # Since we are maximizing, and we already did greedy, we are likely stuck.
        # Local search: try swapping a triple.
        # Or try removing a triple that blocks many additions, then re-greedy.
        
        # Simpler: try to find a triple t not in set that can be added if we remove one existing triple u.
        # This is expensive.
        
        # Alternative: Random restarts with different shuffles are often effective for this size.
        # Let's just do multiple greedy runs with different seeds and return the best.
        
        return current_set

    best_size = 0
    best_set = set()
    
    # Run multiple greedy constructions
    num_runs = 200
    for i in range(num_runs):
        if time.time() > deadline - 5.0:
            break
            
        s = greedy_construction(i)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
        # If we have time, try a simple local improvement on the best set found so far?
        # Maybe not necessary if greedy with randomization is good enough.
        # But let's try a "repair" step: take the best set, try to add triples again in random order.
        # This might find triples that were skipped because they were invalid at the time,
        # but became valid? No, adding triples only makes it harder to add more (monotonic constraint violation).
        # Wait, the constraint is "no K(2,2,2)". Adding a triple can create a K(2,2,2).
        # It never removes a K(2,2,2). So if a triple was invalid, it remains invalid.
        # So re-trying on the same set is useless.
        
        # However, a different order of adding might lead to a different local optimum.
        # That's what the different seeds do.

    # Convert best_set back to list of lists
    result = [list(t) for t in best_set]
    return result

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
