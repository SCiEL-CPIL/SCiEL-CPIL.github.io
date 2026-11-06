import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is formed by three disjoint pairs (a,a'), (b,b'), (c,c')
    # The 8 transversal triples are: (a,b,c), (a,b,c'), (a,b',c), (a,b',c'),
    #                                (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    k222_configs = []
    seen_k222 = set()
    
    for a, b, c in combinations(range(n), 3):
        # Choose 3 more points to form the "other" elements of each pair
        for a2, b2, c2 in combinations(range(n), 3):
            if len({a, b, c, a2, b2, c2}) != 6:
                continue
            # Check they form 3 disjoint pairs
            pairs = [(a, a2), (b, b2), (c, c2)]
            all_points = set()
            for p, q in pairs:
                all_points.add(p)
                all_points.add(q)
            if len(all_points) != 6:
                continue
            
            # Generate the 8 transversal triples
            trans_triples = []
            for x in (a, a2):
                for y in (b, b2):
                    for z in (c, c2):
                        t = tuple(sorted((x, y, z)))
                        if t not in trans_triples:
                            trans_triples.append(t)
            
            if len(trans_triples) == 8:
                key = frozenset(trans_triples)
                if key not in seen_k222:
                    seen_k222.add(key)
                    k222_configs.append(key)
    
    num_k222 = len(k222_configs)
    
    # For each triple, track which K(2,2,2) configs it belongs to
    triple_k222 = [[] for _ in range(num_triples)]
    for ci, config in enumerate(k222_configs):
        for t in config:
            idx = triple_to_idx[t]
            triple_k222[idx].append(ci)
    
    # Greedy construction with bitmask tracking
    # active_k222[ci] = number of triples from config ci that are in our set
    active_k222 = [0] * num_k222
    in_set = [False] * num_triples
    result = []
    
    # Shuffle triples for randomization
    order = list(range(num_triples))
    random.seed(42)
    random.shuffle(order)
    
    for idx in order:
        if time.time() > deadline - 5:
            break
        t = all_triples[idx]
        # Check if adding this triple would complete any K(2,2,2)
        can_add = True
        for ci in triple_k222[idx]:
            if active_k222[ci] == 7:  # Adding this would make it 8
                can_add = False
                break
        if can_add:
            in_set[idx] = True
            result.append(list(t))
            for ci in triple_k222[idx]:
                active_k222[ci] += 1
    
    best_result = list(result)
    best_size = len(result)
    
    # Local search: try "remove 1, add 2" moves
    search_start = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 10 and no_improve_count < max_no_improve:
        improved = False
        
        # Try to find a triple to remove and two to add
        for remove_idx in range(min(200, num_triples)):
            if not in_set[remove_idx]:
                continue
            if time.time() > deadline - 10:
                break
            
            # Temporarily remove
            in_set[remove_idx] = False
            for ci in triple_k222[remove_idx]:
                active_k222[ci] -= 1
            
            # Find two triples to add that don't conflict
            add_candidates = []
            for add_idx in order:
                if in_set[add_idx]:
                    continue
                t = all_triples[add_idx]
                can_add = True
                for ci in triple_k222[add_idx]:
                    if active_k222[ci] == 7:
                        can_add = False
                        break
                if can_add:
                    add_candidates.append(add_idx)
                if len(add_candidates) >= 2:
                    break
            
            if len(add_candidates) >= 2:
                # Try adding the first two candidates
                a1, a2 = add_candidates[0], add_candidates[1]
                
                # Check if both can be added simultaneously
                temp_active = list(active_k222)
                for ci in triple_k222[a1]:
                    temp_active[ci] += 1
                both_ok = True
                for ci in triple_k222[a2]:
                    if temp_active[ci] >= 7:
                        both_ok = False
                        break
                
                if both_ok:
                    # Apply the move
                    in_set[a1] = True
                    in_set[a2] = True
                    result.append(list(all_triples[a1]))
                    result.append(list(all_triples[a2]))
                    for ci in triple_k222[a1]:
                        active_k222[ci] += 1
                    for ci in triple_k222[a2]:
                        active_k222[ci] += 1
                    
                    improved = True
                    no_improve_count = 0
                    if len(result) > best_size:
                        best_size = len(result)
                        best_result = list(result)
                    break
            else:
                # Restore
                in_set[remove_idx] = True
                for ci in triple_k222[remove_idx]:
                    active_k222[ci] += 1
        
        if not improved:
            no_improve_count += 1
    
    return best_result
