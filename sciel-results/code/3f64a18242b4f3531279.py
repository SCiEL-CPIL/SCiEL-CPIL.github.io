"""Rewritten program: greedy construction with local search for max K(2,2,2)-free triple system on 16 points."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_set = set(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560
    
    # Precompute conflict structure:
    # A K(2,2,2) is determined by 3 disjoint pairs (a1,a2), (b1,b2), (c1,c2)
    # The 8 transversal triples are (ai, bj, ck) for i,j,k in {0,1}
    # Two triples conflict if they are both part of some K(2,2,2)
    # Actually, the constraint is: no set of 8 triples forming a K(2,2,2) should all be present
    # So we need to avoid having all 8 transversal triples of any K(2,2,2)
    
    # Let's enumerate all K(2,2,2) configurations
    # A K(2,2,2) is specified by 3 disjoint pairs
    # Number of ways to choose 3 disjoint pairs from 16 points:
    # Choose 6 points, then partition into 3 pairs: C(16,6) * 15 = 8008 * 15 = 120120
    # For each such partition, the 8 transversal triples are determined
    
    # For efficiency, let's think differently.
    # Two triples T1 = {a,b,c} and T2 = {d,e,f} "conflict" in the sense that there exists
    # a K(2,2,2) containing both. But the actual constraint is that all 8 triples of a K(2,2,2)
    # cannot all be present.
    
    # Better approach: for each pair of disjoint triples, check if they can be part of the same K(2,2,2)
    # Actually, let's precompute for each triple which K(2,2,2)s it belongs to.
    
    # Alternative: use a different conflict definition.
    # Two triples are in conflict if they share exactly 1 point and the remaining 4 points can form
    # the other parts of a K(2,2,2). This is getting complex.
    
    # Let me use a simpler but effective approach:
    # Greedy with random ordering, then local search (add/remove/swap)
    
    # For the local search, I need to efficiently check if adding a triple creates a K(2,2,2)
    # A triple T = {a,b,c} is "bad" if there exist two other triples T1, T2 such that
    # {T, T1, T2} are part of a K(2,2,2) where all 8 transversals are present.
    
    # Actually, the simplest check: when we have a set S of triples, S is valid iff
    # for every K(2,2,2) configuration K, not all 8 transversal triples of K are in S.
    
    # Let's precompute all K(2,2,2) configurations and their 8 transversal triples.
    # This might be too many (120120), but let's try to be smart.
    
    # Actually, for the greedy + local search, I can use a different approach:
    # Maintain a count for each K(2,2,2) of how many of its 8 triples are in S.
    # A triple T can be added iff for every K(2,2,2) containing T, the count is < 7
    # (i.e., adding T would make count = 8, which is forbidden).
    
    # But enumerating all K(2,2,2)s is expensive. Let me think of a better way.
    
    # For two triples T1 = {a,b,c} and T2 = {d,e,f} that are disjoint,
    # they can be part of a K(2,2,2) if we can find a third triple T3 = {g,h,i}
    # such that {T1, T2, T3} are 3 of the 8 transversals, and the other 5 can be formed.
    
    # This is getting complex. Let me use a heuristic approach:
    # Greedy construction with random restarts and local search.
    # For validity checking, I'll use a direct check when needed.
    
    def is_valid(triple_list):
        """Check if the list of triples contains no complete K(2,2,2)."""
        ts = set(triple_list)
        # For each pair of disjoint triples, check if they can complete a K(2,2,2)
        # This is O(n^4) which is too slow for 560 triples.
        # Instead, check all K(2,2,2) configurations.
        # A K(2,2,2) is defined by 3 disjoint pairs.
        # For efficiency, iterate over all ways to pick 3 disjoint pairs.
        # But that's 120120 configurations, each with 8 triples to check.
        # 120120 * 8 = ~1M checks, which is feasible.
        
        # Actually, let's precompute all K(2,2,2) configurations once.
        return True  # Placeholder, we'll do better
    
    # Precompute all K(2,2,2) configurations
    # Each is a set of 8 triples
    k222_configs = []
    for six_points in combinations(range(n), 6):
        # Partition 6 points into 3 pairs
        a, b, c, d, e, f = six_points
        # 15 ways to partition 6 items into 3 pairs
        pairings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in pairings:
            # Generate 8 transversal triples
            x1, x2 = p1
            y1, y2 = p2
            z1, z2 = p3
            transversals = set()
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        t = tuple(sorted([x1 if i==0 else x2, y1 if j==0 else y2, z1 if k==0 else z2]))
                        transversals.add(t)
            if len(transversals) == 8:
                k222_configs.append(frozenset(transversals))
    
    # Remove duplicates
    k222_configs = list(set(k222_configs))
    
    # For each triple, which K(2,2,2) configs contain it?
    triple_to_k222 = [[] for _ in range(num_triples)]
    for ki, config in enumerate(k222_configs):
        for t in config:
            ti = triple_index[t]
            triple_to_k222[ti].append(ki)
    
    # Now implement greedy + local search
    best_result = []
    best_size = 0
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Greedy phase: add triples in random order
        order = list(range(num_triples))
        random.shuffle(order)
        
        current_set = set()
        k222_counts = [0] * len(k222_configs)
        
        for ti in order:
            t = all_triples[ti]
            # Check if adding t would complete any K(2,2,2)
            can_add = True
            for ki in triple_to_k222[ti]:
                if k222_counts[ki] == 7:
                    can_add = False
                    break
            if can_add:
                current_set.add(t)
                for ki in triple_to_k222[ti]:
                    k222_counts[ki] += 1
        
        current_size = len(current_set)
        
        # Local search phase
        improved = True
        local_iter = 0
        while improved and time.time() < deadline - 5 and local_iter < 2000:
            improved = False
            local_iter += 1
            
            # Try to add a random triple not in current_set
            if time.time() > deadline - 5:
                break
            
            # Try adding
            for _ in range(50):
                ti = random.randint(0, num_triples - 1)
                t = all_triples[ti]
                if t in current_set:
                    continue
                can_add = True
                for ki in triple_to_k222[ti]:
                    if k222_counts[ki] == 7:
                        can_add = False
                        break
                if can_add:
                    current_set.add(t)
                    for ki in triple_to_k222[ti]:
                        k222_counts[ki] += 1
                    current_size += 1
                    improved = True
                    break
            
            # Try 1-for-1 swap: remove one, add one
            if not improved and current_size > 0:
                for _ in range(30):
                    ti_out = random.randint(0, num_triples - 1)
                    if all_triples[ti_out] not in current_set:
                        continue
                    ti_in = random.randint(0, num_triples - 1)
                    if all_triples[ti_in] in current_set:
                        continue
                    
                    t_out = all_triples[ti_out]
                    t_in = all_triples[ti_in]
                    
                    # Check if removing t_out and adding t_in is valid
                    # First check if t_in can be added (with t_out still present)
                    can_add_in = True
                    for ki in triple_to_k222[ti_in]:
                        if k222_counts[ki] == 7:
                            can_add_in = False
                            break
                    if not can_add_in:
                        continue
                    
                    # After removing t_out, check if t_in can be added
                    # We need to temporarily remove t_out
                    for ki in triple_to_k222[ti_out]:
                        k222_counts[ki] -= 1
                    
                    can_add_in_after = True
                    for ki in triple_to_k222[ti_in]:
                        if k222_counts[ki] == 7:
                            can_add_in_after = False
                            break
                    
                    if can_add_in_after:
                        # Perform the swap
                        current_set.remove(t_out)
                        current_set.add(t_in)
                        for ki in triple_to_k222[ti_in]:
                            k222_counts[ki] += 1
                        improved = True
                        break
                    else:
                        # Restore
                        for ki in triple_to_k222[ti_out]:
                            k222_counts[ki] += 1
        
        if current_size > best_size:
            best_size = current_size
            best_result = list(current_set)
    
    # Convert to list of lists
    return [list(t) for t in best_result]
