import itertools
import random

def construct():
    # Precompute all triples
    vertices = list(range(16))
    all_triples = list(itertools.combinations(vertices, 3))
    # Convert to tuples for hashing, keep original list structure for return
    
    # Precompute conflicts for each triple
    # conflicts[t] is a list of frozensets of 7 triples
    # If any of these sets is a subset of current S, adding t creates a K(2,2,2)
    conflicts = {}
    
    # To speed up, we can map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    for t in all_triples:
        u, v, w = t
        remaining = [x for x in vertices if x not in t]
        conf_list = []
        # We need to pick 3 distinct vertices from remaining and assign to the 3 "slots"
        # The slots correspond to the parts containing u, v, w.
        # Since the parts are unlabeled in the final K(2,2,2) definition, 
        # but the structure is defined by the 6 vertices and the 8 edges.
        # The 8 edges are all transversals.
        # If we fix t = {u,v,w}, the other 3 vertices are x,y,z.
        # The 8 edges are {u,v,w}, {u,v,z}, {u,y,w}, {u,y,z}, {x,v,w}, {x,v,z}, {x,y,w}, {x,y,z}.
        # Wait, this assumes u,v,w are in distinct parts.
        # In a K(2,2,2), the 6 vertices are partitioned into 3 pairs.
        # t must contain exactly one vertex from each pair.
        # So if we pick x,y,z, we must ensure that {u,x}, {v,y}, {w,z} are the pairs?
        # No, the pairs can be any partition of {u,v,w,x,y,z} into 3 pairs such that t intersects each in 1 vertex.
        # Given t={u,v,w} and x,y,z, how many such partitions?
        # We need to pair u with one of {x,y,z}, v with another, w with the last.
        # There are 3! = 6 ways.
        # So for each combination of 3 vertices {x,y,z} from remaining, there are 6 potential K(2,2,2)s.
        
        # Let's iterate over permutations of 3 distinct vertices from remaining
        # There are P(13,3) = 1716 such ordered tuples (x,y,z)
        # For each (x,y,z), we consider the 6 pairings:
        # 1. (u,x), (v,y), (w,z) -> other 7 edges: 
        #    {u,y,w}, {u,y,z}, {x,v,w}, {x,v,z}, {x,y,w}, {x,y,z}, {u,v,z} ? 
        #    Let's derive carefully.
        #    Parts: A={u,x}, B={v,y}, C={w,z}.
        #    Edges: 
        #    1. {u,v,w} (t)
        #    2. {u,v,z}
        #    3. {u,y,w}
        #    4. {u,y,z}
        #    5. {x,v,w}
        #    6. {x,v,z}
        #    7. {x,y,w}
        #    8. {x,y,z}
        #    So the other 7 are: {u,v,z}, {u,y,w}, {u,y,z}, {x,v,w}, {x,v,z}, {x,y,w}, {x,y,z}.
        
        # We can just generate these 7 sets for each of the 6 permutations of (x,y,z) assignment to (A,B,C) slots relative to (u,v,w).
        # Actually, iterating over permutations of (x,y,z) from remaining covers all assignments if we fix the pairing logic.
        # Let's just iterate over all permutations of 3 distinct elements from remaining.
        # There are 13*12*11 = 1716.
        # For each (x,y,z), we assume x pairs with u, y with v, z with w? 
        # No, that's only 1 of the 6 pairings.
        # We need to check all 6 pairings for the set {x,y,z}.
        # Or, we can just iterate over all 1716 permutations of (x,y,z) and for each, consider the 6 ways to pair?
        # No, that would be overcounting or undercounting.
        
        # Simpler: Iterate over all combinations of 3 vertices {x,y,z} from remaining (286 combos).
        # For each combo, there are 6 ways to pair them with u,v,w.
        # Total 286 * 6 = 1716 conflict sets per triple.
        
        for combo in itertools.combinations(remaining, 3):
            x, y, z = combo
            # 6 permutations of assigning x,y,z to pair with u,v,w
            # Let perm be a tuple (p1, p2, p3) where p1 pairs with u, p2 with v, p3 with w
            for perm in itertools.permutations(combo):
                px, py, pz = perm
                # Parts: {u,px}, {v,py}, {w,pz}
                # Other 7 triples:
                others = [
                    frozenset((u, v, pz)),
                    frozenset((u, py, w)),
                    frozenset((u, py, pz)),
                    frozenset((px, v, w)),
                    frozenset((px, v, pz)),
                    frozenset((px, py, w)),
                    frozenset((px, py, pz))
                ]
                # Convert to tuples for consistency with all_triples keys
                others_tuples = [tuple(sorted(o)) for o in others]
                conf_list.append(frozenset(others_tuples))
        
        conflicts[t] = conf_list

    best_size = 0
    best_set = set()
    
    # Run multiple random restarts
    # 560 triples. Greedy is O(N^2 * check_cost). 
    # Check cost is 1716 subset checks.
    # Total ops ~ 560 * 560 * 1716 ~ 500M? 
    # Wait, in the loop, for each candidate t, we check against current S.
    # S grows. 
    # Average S size ~ 200.
    # But the check is `if cs.issubset(S)`. 
    # `issubset` on a set of 7 elements is fast.
    # 1716 checks per candidate.
    # 560 candidates.
    # 560 * 1716 = ~1M operations per restart.
    # We can do 100 restarts easily.
    
    for _ in range(200):
        random.shuffle(all_triples)
        current_set = set()
        for t in all_triples:
            # Check if t can be added
            safe = True
            for cs in conflicts[t]:
                if cs.issubset(current_set):
                    safe = False
                    break
            if safe:
                current_set.add(t)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    # Return list of lists
    return [list(t) for t in best_set]
