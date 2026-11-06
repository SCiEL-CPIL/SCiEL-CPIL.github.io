import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    points = list(range(n))
    
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Bitmask representation
    # 560 bits = 9 uint64s (or just use Python ints, which are arbitrary precision)
    # Python ints are efficient for bit operations.
    
    # Pre-compute neighbor masks
    # triple_to_neighbors[t] = list of masks, where each mask represents the other 7 triples in a config containing t
    
    triple_to_neighbors = {t: [] for t in all_triples}
    
    # Iterate over 6-point subsets
    for combo in itertools.combinations(points, 6):
        a, b, c, d, e, f = combo
        matchings = []
        # Generate 15 matchings as before
        # a-b
        rem = [c, d, e, f]
        matchings.append([(a, b), (rem[0], rem[1]), (rem[2], rem[3])])
        matchings.append([(a, b), (rem[0], rem[2]), (rem[1], rem[3])])
        matchings.append([(a, b), (rem[0], rem[3]), (rem[1], rem[2])])
        # a-c
        rem = [b, d, e, f]
        matchings.append([(a, c), (rem[0], rem[1]), (rem[2], rem[3])])
        matchings.append([(a, c), (rem[0], rem[2]), (rem[1], rem[3])])
        matchings.append([(a, c), (rem[0], rem[3]), (rem[1], rem[2])])
        # a-d
        rem = [b, c, e, f]
        matchings.append([(a, d), (rem[0], rem[1]), (rem[2], rem[3])])
        matchings.append([(a, d), (rem[0], rem[2]), (rem[1], rem[3])])
        matchings.append([(a, d), (rem[0], rem[3]), (rem[1], rem[2])])
        # a-e
        rem = [b, c, d, f]
        matchings.append([(a, e), (rem[0], rem[1]), (rem[2], rem[3])])
        matchings.append([(a, e), (rem[0], rem[2]), (rem[1], rem[3])])
        matchings.append([(a, e), (rem[0], rem[3]), (rem[1], rem[2])])
        # a-f
        rem = [b, c, d, e]
        matchings.append([(a, f), (rem[0], rem[1]), (rem[2], rem[3])])
        matchings.append([(a, f), (rem[0], rem[2]), (rem[1], rem[3])])
        matchings.append([(a, f), (rem[0], rem[3]), (rem[1], rem[2])])
        
        for matching in matchings:
            p1, p2, p3 = matching
            t1 = tuple(sorted((p1[0], p2[0], p3[0])))
            t2 = tuple(sorted((p1[0], p2[0], p3[1])))
            t3 = tuple(sorted((p1[0], p2[1], p3[0])))
            t4 = tuple(sorted((p1[0], p2[1], p3[1])))
            t5 = tuple(sorted((p1[1], p2[0], p3[0])))
            t6 = tuple(sorted((p1[1], p2[0], p3[1])))
            t7 = tuple(sorted((p1[1], p2[1], p3[0])))
            t8 = tuple(sorted((p1[1], p2[1], p3[1])))
            
            config = [t1, t2, t3, t4, t5, t6, t7, t8]
            
            for t in config:
                idx_t = triple_to_idx[t]
                # Create mask for the other 7
                mask = 0
                for other in config:
                    if other != t:
                        idx_other = triple_to_idx[other]
                        mask |= (1 << idx_other)
                triple_to_neighbors[t].append(mask)
                
    best_mask = 0
    best_count = 0
    
    # Greedy with random restarts
    while time.time() < deadline:
        random.shuffle(all_triples)
        current_mask = 0
        count = 0
        
        for t in all_triples:
            idx_t = triple_to_idx[t]
            # Check if adding t creates a forbidden config
            # If any neighbor mask is a subset of current_mask, then it's invalid
            is_valid = True
            for n_mask in triple_to_neighbors[t]:
                if (current_mask & n_mask) == n_mask:
                    is_valid = False
                    break
            
            if is_valid:
                current_mask |= (1 << idx_t)
                count += 1
                
        if count > best_count:
            best_count = count
            best_mask = current_mask
            
    # Convert best_mask back to list of triples
    result = []
    for i, t in enumerate(all_triples):
        if (best_mask >> i) & 1:
            result.append(list(t))
            
    return result
