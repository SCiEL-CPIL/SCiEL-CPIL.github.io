"""Construct a large triple system on 16 points avoiding K(2,2,2)."""
import time
import itertools
import random

def _all_triples():
    return [list(t) for t in itertools.combinations(range(16), 3)]

def _triple_to_set(t):
    return frozenset(t)

def _find_k222(triples_set, triples_list):
    """Check if there exists a K(2,2,2): three disjoint pairs whose 8 transversal triples are all present.
    Returns True if found, False otherwise."""
    # A K(2,2,2) consists of 3 disjoint pairs: {a,b}, {c,d}, {e,f}
    # and requires all 8 triples: {a,c,e}, {a,c,f}, {a,d,e}, {a,d,f},
    #                              {b,c,e}, {b,c,f}, {b,d,e}, {b,d,f}
    # We need to check if any such configuration exists.
    
    # Enumerate all ways to pick 3 disjoint pairs from 16 points.
    # 16 points, pick 6, partition into 3 pairs: C(16,6) * 15 = 8008 * 15 = 120120
    # For each, check 8 triples in the set.
    
    # Precompute: for speed, use the set of frozensets.
    for pts in itertools.combinations(range(16), 6):
        # Generate all perfect matchings of 6 points (15 matchings)
        # 0-1, 2-3, 4-5
        # 0-1, 2-4, 3-5
        # 0-1, 2-5, 3-4
        # 0-2, 1-3, 4-5
        # 0-2, 1-4, 3-5
        # 0-2, 1-5, 3-4
        # 0-3, 1-2, 4-5
        # 0-3, 1-4, 2-5
        # 0-3, 1-5, 2-4
        # 0-4, 1-2, 3-5
        # 0-4, 1-3, 2-5
        # 0-4, 1-5, 2-3
        # 0-5, 1-2, 3-4
        # 0-5, 1-3, 2-4
        # 0-5, 1-4, 2-3
        # Actually there are 15 perfect matchings of K6.
        a, b, c, d, e, f = pts
        matchings = [
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
        for p1, p2, p3 in matchings:
            # Check all 8 transversal triples
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            if (frozenset((x1,x2,x3)) in triples_set and
                frozenset((x1,x2,y3)) in triples_set and
                frozenset((x1,y2,x3)) in triples_set and
                frozenset((x1,y2,y3)) in triples_set and
                frozenset((y1,x2,x3)) in triples_set and
                frozenset((y1,x2,y3)) in triples_set and
                frozenset((y1,y2,x3)) in triples_set and
                frozenset((y1,y2,y3)) in triples_set):
                return True
    return False

def _random_greedy(n_triples_target, time_limit):
    """Random greedy: add triples in random order if they don't create K(2,2,2)."""
    triples = _all_triples()
    random.shuffle(triples)
    
    selected = []
    selected_set = set()
    
    start = time.time()
    for t in triples:
        if time.time() - start > time_limit:
            break
        ts = frozenset(t)
        if ts in selected_set:
            continue
        # Check if adding this triple creates a K(2,2,2)
        # We need to check if there exist 3 disjoint pairs such that this triple is one of the 8
        # and the other 7 are already in selected_set.
        # This is expensive. Let's use a simpler check: just add it and check the whole thing? No, too slow.
        
        # Alternative: for each potential K(2,2,2) that this triple could be part of,
        # check if the other 7 triples are present.
        # A triple {a,b,c} can be part of a K(2,2,2) in various ways.
        # In a K(2,2,2) with pairs {x1,y1},{x2,y2},{x3,y3}, the 8 triples are all combinations.
        # {a,b,c} must be one of these. So a,b,c must be one from each pair.
        # WLOG, a is from pair1, b from pair2, c from pair3.
        # Then the other 7 triples are:
        # {y1,b,c}, {a,y2,c}, {a,b,y3}, {y1,y2,c}, {y1,b,y3}, {a,y2,y3}, {y1,y2,y3}
        
        a, b, c = t
        # We need to find y1, y2, y3 such that:
        # {y1,b,c}, {a,y2,c}, {a,b,y3}, {y1,y2,c}, {y1,b,y3}, {a,y2,y3}, {y1,y2,y3} are all in selected_set
        # and y1 != a, y2 != b, y3 != c, and all 6 points are distinct.
        
        # This is still complex. Let's try a different approach:
        # Just check _find_k222 on the new set. But that's O(120120 * 8) per addition, too slow.
        
        # Better: for each pair (y1, y2) where y1 != a, y2 != b, y1 != y2, y1 != b, y1 != c, y2 != a, y2 != c:
        #   We need y3 such that {a,b,y3}, {y1,y2,c}, {y1,b,y3}, {a,y2,y3}, {y1,y2,y3} are in selected_set
        #   and {y1,b,c} is in selected_set.
        #   Also y3 must be distinct from a,b,c,y1,y2.
        
        # Let's precompute for speed.
        found = False
        for y1 in range(16):
            if y1 == a or y1 == b or y1 == c:
                continue
            t1 = frozenset((y1, b, c))
            if t1 not in selected_set:
                continue
            for y2 in range(16):
                if y2 == a or y2 == b or y2 == c or y2 == y1:
                    continue
                t2 = frozenset((a, y2, c))
                if t2 not in selected_set:
                    continue
                t3 = frozenset((y1, y2, c))
                if t3 not in selected_set:
                    continue
                for y3 in range(16):
                    if y3 == a or y3 == b or y3 == c or y3 == y1 or y3 == y2:
                        continue
                    t4 = frozenset((a, b, y3))
                    if t4 not in selected_set:
                        continue
                    t5 = frozenset((y1, b, y3))
                    if t5 not in selected_set:
                        continue
                    t6 = frozenset((a, y2, y3))
                    if t6 not in selected_set:
                        continue
                    t7 = frozenset((y1, y2, y3))
                    if t7 not in selected_set:
                        continue
                    found = True
                    break
                if found:
                    break
            if found:
                break
        if not found:
            selected.append(t)
            selected_set.add(ts)
    
    return selected

def _local_search(initial, time_limit):
    """Try to improve by adding more triples via local search."""
    selected = list(initial)
    selected_set = set(frozenset(t) for t in selected)
    
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        candidates = [t for t in _all_triples() if frozenset(t) not in selected_set]
        random.shuffle(candidates)
        for t in candidates:
            if time.time() - start > time_limit:
                break
            ts = frozenset(t)
            a, b, c = t
            # Check if adding t creates K(2,2,2)
            found = False
            for y1 in range(16):
                if y1 == a or y1 == b or y1 == c:
                    continue
                if frozenset((y1, b, c)) not in selected_set:
                    continue
                for y2 in range(16):
                    if y2 == a or y2 == b or y2 == c or y2 == y1:
                        continue
                    if frozenset((a, y2, c)) not in selected_set:
                        continue
                    if frozenset((y1, y2, c)) not in selected_set:
                        continue
                    for y3 in range(16):
                        if y3 == a or y3 == b or y3 == c or y3 == y1 or y3 == y2:
                            continue
                        if frozenset((a, b, y3)) not in selected_set:
                            continue
                        if frozenset((y1, b, y3)) not in selected_set:
                            continue
                        if frozenset((a, y2, y3)) not in selected_set:
                            continue
                        if frozenset((y1, y2, y3)) not in selected_set:
                            continue
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if not found:
                selected.append(t)
                selected_set.add(ts)
                improved = True
    
    return selected

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    # Start with the seed: all triples through point 0
    # That's C(15,2) = 105 triples
    best = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    # Try random greedy multiple times
    attempts = 0
    while time.time() < deadline - 30:
        attempts += 1
        random.seed(attempts * 12345 + 67890)
        result = _random_greedy(560, 10)
        if len(result) > len(best):
            best = result
            # Try local search to improve
            if time.time() < deadline - 20:
                improved = _local_search(best, deadline - time.time() - 5)
                if len(improved) > len(best):
                    best = improved
    
    return best
