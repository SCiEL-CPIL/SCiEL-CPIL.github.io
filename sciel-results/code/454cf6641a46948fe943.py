import random
import itertools
import time
import numpy as np

def construct():
    n = 16
    points = list(range(n))
    
    all_triples = []
    triple_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx = len(all_triples)
                all_triples.append((i, j, k))
                triple_to_idx[(i, j, k)] = idx
                
    num_triples = len(all_triples)
    
    # Precompute K(2,2,2) configurations
    # conflict_map: list of lists of k222 indices
    conflict_map = [[] for _ in range(num_triples)]
    
    # Generate matchings of 6 points
    # We can precompute the 15 matchings of (0,1,2,3,4,5) as patterns
    # and apply them to any combo of 6 points.
    
    # Patterns of pairs for indices 0..5
    # We need all 15 perfect matchings.
    # We can generate them once for the canonical set (0,1,2,3,4,5)
    # and then map them to the actual points.
    
    def get_matchings_6():
        # Returns list of 15 lists of 3 pairs (tuples of indices 0..5)
        matchings = []
        def rec(rem, current):
            if not rem:
                matchings.append(current)
                return
            first = rem[0]
            rest = rem[1:]
            for i in range(len(rest)):
                pair = (first, rest[i])
                new_rem = rest[:i] + rest[i+1:]
                rec(new_rem, current + [pair])
        rec(list(range(6)), [])
        return matchings

    canonical_matchings = get_matchings_6()
    
    for combo in itertools.combinations(points, 6):
        # combo is a tuple of 6 points
        # Map canonical indices 0..5 to combo[0]..combo[5]
        # For each canonical matching, generate the 8 triples
        for m in canonical_matchings:
            # m is list of 3 pairs of canonical indices
            p1, p2, p3 = m
            # Get actual points
            a1, a2 = combo[p1[0]], combo[p1[1]]
            b1, b2 = combo[p2[0]], combo[p2[1]]
            c1, c2 = combo[p3[0]], combo[p3[1]]
            
            # 8 triples
            t1 = tuple(sorted(a1, b1, c1))
            t2 = tuple(sorted(a1, b1, c2))
            t3 = tuple(sorted(a1, b2, c1))
            t4 = tuple(sorted(a1, b2, c2))
            t5 = tuple(sorted(a2, b1, c1))
            t6 = tuple(sorted(a2, b1, c2))
            t7 = tuple(sorted(a2, b2, c1))
            t8 = tuple(sorted(a2, b2, c2))
            
            triples_in_k222 = [t1, t2, t3, t4, t5, t6, t7, t8]
            idxs = [triple_to_idx[t] for t in triples_in_k222]
            k222_idx = len(conflict_map[0]) # This is wrong, need a global counter
            # We need to store k222 indices properly.
            # Let's just append to conflict_map directly and keep a count
            # But we need the same k222_idx for all 8 triples.
            # So we need a global list of k222s or just a counter.
            pass

    # Let's restructure precomputation to be cleaner
    conflict_map = [[] for _ in range(num_triples)]
    k222_count = 0
    for combo in itertools.combinations(points, 6):
        for m in canonical_matchings:
            p1, p2, p3 = m
            a1, a2 = combo[p1[0]], combo[p1[1]]
            b1, b2 = combo[p2[0]], combo[p2[1]]
            c1, c2 = combo[p3[0]], combo[p3[1]]
            
            t1 = tuple(sorted(a1, b1, c1))
            t2 = tuple(sorted(a1, b1, c2))
            t3 = tuple(sorted(a1, b2, c1))
            t4 = tuple(sorted(a1, b2, c2))
            t5 = tuple(sorted(a2, b1, c1))
            t6 = tuple(sorted(a2, b1, c2))
            t7 = tuple(sorted(a2, b2, c1))
            t8 = tuple(sorted(a2, b2, c2))
            
            idxs = [triple_to_idx[t] for t in (t1, t2, t3, t4, t5, t6, t7, t8)]
            for idx in idxs:
                conflict_map[idx].append(k222_count)
            k222_count += 1
            
    num_k222 = k222_count
    
    # Convert conflict_map to numpy arrays for fast access
    conflict_map_np = []
    for lst in conflict_map:
        conflict_map_np.append(np.array(lst, dtype=np.int32))
        
    best_solution = []
    best_size = 0
    
    start_time = time.time()
    time_limit = 100
    
    for restart in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(restart)
        order = list(range(num_triples))
        random.shuffle(order)
        
        selected = set()
        k222_counts = np.zeros(num_k222, dtype=np.int32)
        
        for t_idx in order:
            idx_array = conflict_map_np[t_idx]
            # Check if any count is 7
            if np.any(k222_counts[idx_array] == 7):
                continue
                
            selected.add(t_idx)
            k222_counts[idx_array] += 1
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_solution = [list(all_triples[i]) for i in selected]
            
    return best_solution
