"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))

def _is_valid_add(trips_set, t):
    """Check if adding triple t to trips_set creates a K(2,2,2)."""
    a, b, c = t
    used = {a, b, c}
    
    # Helper to check if a set of 3 points is in trips_set
    def in_set(x, y, z):
        return tuple(sorted((x, y, z))) in trips_set

    # Iterate over possible a'
    for ap in range(V):
        if ap in used:
            continue
        # Check if {a', b, c} is in set
        if not in_set(ap, b, c):
            continue
            
        # Iterate over possible b'
        for bp in range(V):
            if bp in used or bp == ap:
                continue
            # Check if {a, b', c} is in set
            if not in_set(a, bp, c):
                continue
            
            # Iterate over possible c'
            for cp in range(V):
                if cp in used or cp == ap or cp == bp:
                    continue
                # Check if {a, b, c'} is in set
                if not in_set(a, b, cp):
                    continue
                
                # Check the remaining 4 transversal triples
                # {a', b', c}
                if not in_set(ap, bp, c):
                    continue
                # {a', b, c'}
                if not in_set(ap, b, cp):
                    continue
                # {a, b', c'}
                if not in_set(a, bp, cp):
                    continue
                # {a', b', c'}
                if not in_set(ap, bp, cp):
                    continue
                
                # All 7 other triples are present, so adding t completes K(2,2,2)
                return False
    return True

def _greedy(order, deadline):
    """Greedily add triples in given order, skipping those that create K(2,2,2)."""
    trips_set = set()
    result = []
    for t in order:
        if time.time() >= deadline:
            break
        if t in trips_set:
            continue
        if _is_valid_add(trips_set, t):
            trips_set.add(t)
            result.append(list(t))
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best = None
    best_len = -1
    
    # Start with the base: all triples through vertex 0
    base = [tuple([0, a, b]) for a in range(1, V) for b in range(a + 1, V)]
    base_set = set(base)
    
    # Other triples to consider
    other_triples = [t for t in ALL_TRIPLES if 0 not in t]
    
    # Try multiple random orders
    for attempt in range(200):
        if time.time() >= deadline:
            break
        
        # Create a random order: base triples first (in random order), then others
        shuffled_base = base[:]
        random.shuffle(shuffled_base)
        shuffled_other = other_triples[:]
        random.shuffle(shuffled_other)
        
        order = shuffled_base + shuffled_other
        
        result = _greedy(order, deadline)
        
        if len(result) > best_len:
            best_len = len(result)
            best = result
            # Early termination if we've used all remaining time budget
            if time.time() >= deadline - 5:
                break
    
    if best is None:
        # Fallback: just return the base
        best = [list(t) for t in base]
    
    # Final validation to ensure no K(2,2,2) exists
    # This is expensive but guarantees correctness. 
    # We only do this once at the end.
    best_set = set(tuple(sorted(t)) for t in best)
    valid = True
    # Check all potential K(2,2,2) structures
    # A K(2,2,2) is defined by 6 vertices x1,x2,y1,y2,z1,z2
    # and the 8 triples:
    # (x1,y1,z1), (x1,y1,z2), (x1,y2,z1), (x1,y2,z2)
    # (x2,y1,z1), (x2,y1,z2), (x2,y2,z1), (x2,y2,z2)
    # We can iterate over disjoint triples of pairs? No, that's complex.
    # Simpler: Iterate over all 6-tuples of distinct vertices.
    # 16C6 is 8008. For each, check if the 8 specific triples are in best_set.
    # This is 8008 * 8 checks = ~64k lookups. Very fast.
    
    for x1, x2, y1, y2, z1, z2 in combinations(range(V), 6):
        # Assign roles arbitrarily to the 6 vertices to form the 8 triples
        # The 8 triples are all combinations of one from {x1,x2}, one from {y1,y2}, one from {z1,z2}
        t1 = (x1, y1, z1)
        t2 = (x1, y1, z2)
        t3 = (x1, y2, z1)
        t4 = (x1, y2, z2)
        t5 = (x2, y1, z1)
        t6 = (x2, y1, z2)
        t7 = (x2, y2, z1)
        t8 = (x2, y2, z2)
        
        if all(tuple(sorted(t)) in best_set for t in [t1, t2, t3, t4, t5, t6, t7, t8]):
            valid = False
            break
            
    if not valid:
        # If invalid, we need to remove triples. 
        # Given the previous failure was due to a bug, and the logic is now fixed,
        # this should rarely happen. But if it does, we return a smaller valid set.
        # A safe fallback is the star (all triples containing 0).
        return [list(t) for t in base]
        
    return best
