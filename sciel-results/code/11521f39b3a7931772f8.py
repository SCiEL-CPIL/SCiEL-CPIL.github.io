import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)

# Map triple to index
triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

def construct():
    start = time.time()
    deadline = start + 110  # Safety margin
    
    best_count = 0
    best_trips = set()
    
    # Baseline: all triples containing 0
    baseline = {tuple(sorted((0, a, b))) for a in range(1, V) for b in range(a+1, V)}
    best_count = len(baseline)
    best_trips = set(baseline)
    
    # Precompute lists of vertices excluding a specific triple's vertices
    # For a triple (a,b,c), the available a' are V \ {a,b,c}
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        presence = [False] * N_TRIPLES
        count = 0
        
        for idx in order:
            if time.time() >= deadline:
                break
            
            a, b, c = ALL_TRIPLES[idx]
            
            # Check for K(2,2,2) conflict
            conflict = False
            
            # Iterate over possible a'
            # a' must be distinct from a, b, c
            for a_p in range(V):
                if a_p == a or a_p == b or a_p == c:
                    continue
                
                # Check if (a_p, b, c) is present
                # Need to sort (a_p, b, c) to get index
                # To avoid sorting overhead, we can precompute or use a fast min/max
                # Since V=16, sorting 3 elements is fast, but we can do better.
                # Let's just use a helper function or inline min/max logic.
                x, y, z = a_p, b, c
                if x > y: x, y = y, x
                if y > z: y, z = z, y
                if x > y: x, y = y, x
                t1_idx = triple_to_idx[(x, y, z)]
                
                if not presence[t1_idx]:
                    continue
                
                # Now we need b' and c'
                # b', c' distinct from a, b, c, a_p
                # And distinct from each other
                # Check remaining 4 conditions for each pair (b_p, c_p)
                
                # Optimization: Precompute the list of valid b' candidates for this a_p?
                # b' must satisfy:
                # 1. {a, b', c} in trips
                # 2. {a_p, b', c} in trips
                # c' must satisfy:
                # 1. {a, b, c'} in trips
                # 2. {a_p, b, c'} in trips
                # And for the pair (b', c'):
                # 1. {a, b', c'} in trips
                # 2. {a_p, b', c'} in trips
                
                # Let's find valid b' candidates
                valid_bp = []
                for b_p in range(V):
                    if b_p in (a, b, c, a_p):
                        continue
                    # Check {a, b_p, c}
                    x, y, z = a, b_p, c
                    if x > y: x, y = y, x
                    if y > z: y, z = z, y
                    if x > y: x, y = y, x
                    if not presence[triple_to_idx[(x, y, z)]]:
                        continue
                    # Check {a_p, b_p, c}
                    x, y, z = a_p, b_p, c
                    if x > y: x, y = y, x
                    if y > z: y, z = z, y
                    if x > y: x, y = y, x
                    if not presence[triple_to_idx[(x, y, z)]]:
                        continue
                    valid_bp.append(b_p)
                    
                if not valid_bp:
                    continue
                
                # Let's find valid c' candidates
                valid_cp = []
                for c_p in range(V):
                    if c_p in (a, b, c, a_p):
                        continue
                    # Check {a, b, c_p}
                    x, y, z = a, b, c_p
                    if x > y: x, y = y, x
                    if y > z: y, z = z, y
                    if x > y: x, y = y, x
                    if not presence[triple_to_idx[(x, y, z)]]:
                        continue
                    # Check {a_p, b, c_p}
                    x, y, z = a_p, b, c_p
                    if x > y: x, y = y, x
                    if y > z: y, z = z, y
                    if x > y: x, y = y, x
                    if not presence[triple_to_idx[(x, y, z)]]:
                        continue
                    valid_cp.append(c_p)
                    
                if not valid_cp:
                    continue
                
                # Now check pairs (b_p, c_p) from valid_bp and valid_cp
                # Must have b_p != c_p
                found_conflict = False
                for b_p in valid_bp:
                    for c_p in valid_cp:
                        if b_p == c_p:
                            continue
                        # Check {a, b_p, c_p}
                        x, y, z = a, b_p, c_p
                        if x > y: x, y = y, x
                        if y > z: y, z = z, y
                        if x > y: x, y = y, x
                        if not presence[triple_to_idx[(x, y, z)]]:
                            continue
                        # Check {a_p, b_p, c_p}
                        x, y, z = a_p, b_p, c_p
                        if x > y: x, y = y, x
                        if y > z: y, z = z, y
                        if x > y: x, y = y, x
                        if not presence[triple_to_idx[(x, y, z)]]:
                            continue
                        
                        found_conflict = True
                        break
                    if found_conflict:
                        break
                if found_conflict:
                    conflict = True
                    break
            
            if not conflict:
                presence[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            best_trips = set(ALL_TRIPLES[i] for i in range(N_TRIPLES) if presence[i])
    
    return [list(t) for t in best_trips]
