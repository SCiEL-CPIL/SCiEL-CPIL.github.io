import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # All C(16,3) = 560 triples
    all_triples = list(combinations(range(16), 3))
    triple_set = set(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is defined by three disjoint pairs
    # For each set of 6 points, there are 15 ways to partition into 3 pairs
    # For each such partition, the 8 transversals form a K(2,2,2)
    
    # Let's enumerate all K(2,2,2) configurations
    # Each is a frozenset of 8 triples
    k222_configs = []
    
    for six_points in combinations(range(16), 6):
        # All ways to partition 6 points into 3 pairs
        # Fix first point, pair it with one of the other 5
        p = list(six_points)
        for i in range(1, 6):
            pair1 = (p[0], p[i])
            remaining = [p[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (remaining[0], remaining[j])
                remaining2 = [remaining[k] for k in range(1, 4) if k != j]
                pair3 = (remaining2[0], remaining2[1])
                
                # The 8 transversals
                a1, a2 = pair1
                b1, b2 = pair2
                c1, c2 = pair3
                transversals = [
                    (a1, b1, c1), (a1, b1, c2), (a1, b2, c1), (a1, b2, c2),
                    (a2, b1, c1), (a2, b1, c2), (a2, b2, c1), (a2, b2, c2)
                ]
                # Sort each triple for consistency
                transversals = [tuple(sorted(t)) for t in transversals]
                k222_configs.append(frozenset(transversals))
    
    # Deduplicate
    k222_configs = list(set(k222_configs))
    num_k222 = len(k222_configs)
    
    # For each triple, which K(2,2,2) configs does it belong to?
    triple_to_k222 = {t: [] for t in all_triples}
    for idx, config in enumerate(k222_configs):
        for t in config:
            triple_to_k222[t].append(idx)
    
    # Start with all triples present
    present = [True] * 560
    present_count = 560
    
    # Greedy removal: while there's a K(2,2,2) fully present, remove the best triple
    # A K(2,2,2) is "active" if all 8 of its triples are present
    
    def find_active_k222():
        for idx, config in enumerate(k222_configs):
            if all(present[triple_index[t]] for t in config):
                return idx
        return -1
    
    # Initial: find and break all K(2,2,2)s
    while time.time() < deadline:
        active_idx = find_active_k222()
        if active_idx == -1:
            break
        
        # Find the triple in this K(2,2,2) that appears in the most active K(2,2,2)s
        # to maximize the number of K(2,2,2)s broken
        best_triple = None
        best_score = -1
        for t in k222_configs[active_idx]:
            ti = triple_index[t]
            if not present[ti]:
                continue
            # Count how many active K(2,2,2)s contain this triple
            count = 0
            for k222_idx in triple_to_k222[t]:
                config = k222_configs[k222_idx]
                if all(present[triple_index[tt]] for tt in config):
                    count += 1
            if count > best_score:
                best_score = count
                best_triple = ti
        
        if best_triple is not None:
            present[best_triple] = False
            present_count -= 1
    
    # Now do local improvement: try to add back triples
    # Try adding each removed triple and check if it creates a new K(2,2,2)
    removed = [i for i in range(560) if not present[i]]
    
    # Randomize order for multiple passes
    random.seed(42)
    
    for iteration in range(20):
        if time.time() >= deadline:
            break
        random.shuffle(removed)
        improved = False
        for i in removed:
            if time.time() >= deadline:
                break
            if present[i]:
                continue
            # Try adding triple i
            present[i] = True
            # Check if any K(2,2,2) becomes fully present
            creates_k222 = False
            for k222_idx in triple_to_k222[all_triples[i]]:
                config = k222_configs[k222_idx]
                if all(present[triple_index[t]] for t in config):
                    creates_k222 = True
                    break
            if creates_k222:
                present[i] = False
            else:
                removed.remove(i)
                present_count += 1
                improved = True
    
    result = [list(all_triples[i]) for i in range(560) if present[i]]
    return result
