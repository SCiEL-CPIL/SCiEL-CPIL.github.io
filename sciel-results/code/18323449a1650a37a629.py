import random
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_to_id = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute partner masks for each triple
    # partner_masks[i] contains masks for each K(2,2,2) that triple i could complete
    partner_masks = [[] for _ in range(num_triples)]
    
    for i, t in enumerate(all_triples):
        a, b, c = t
        others = [x for x in range(n) if x not in (a,b,c)]
        for a_prime in others:
            for b_prime in others:
                if b_prime == a_prime:
                    continue
                for c_prime in others:
                    if c_prime == a_prime or c_prime == b_prime:
                        continue
                    # The 7 partners
                    t1 = tuple(sorted((a,b,c_prime)))
                    t2 = tuple(sorted((a_prime,b,c)))
                    t3 = tuple(sorted((a_prime,b,c_prime)))
                    t4 = tuple(sorted((a,b_prime,c)))
                    t5 = tuple(sorted((a,b_prime,c_prime)))
                    t6 = tuple(sorted((a_prime,b_prime,c)))
                    t7 = tuple(sorted((a_prime,b_prime,c_prime)))
                    
                    mask = 0
                    ids = [triple_to_id[t1], triple_to_id[t2], triple_to_id[t3], 
                           triple_to_id[t4], triple_to_id[t5], triple_to_id[t6], triple_to_id[t7]]
                    
                    for id_ in ids:
                        mask |= (1 << id_)
                    partner_masks[i].append(mask)
    
    best_present_mask = 0
    max_count = 0
    
    # Run multiple random greedy constructions
    for seed in range(50):
        random.seed(seed)
        shuffled_indices = list(range(num_triples))
        random.shuffle(shuffled_indices)
        
        present = 0
        count = 0
        for i in shuffled_indices:
            valid = True
            for mask in partner_masks[i]:
                if (present & mask) == mask:
                    valid = False
                    break
            if valid:
                present |= (1 << i)
                count += 1
        
        if count > max_count:
            max_count = count
            best_present_mask = present
            
    # Local search: try to add any remaining valid triples
    present = best_present_mask
    improved = True
    while improved:
        improved = False
        for i in range(num_triples):
            if not ((present >> i) & 1):
                valid = True
                for mask in partner_masks[i]:
                    if (present & mask) == mask:
                        valid = False
                        break
                if valid:
                    present |= (1 << i)
                    improved = True
                    
    result = []
    for i in range(num_triples):
        if (present >> i) & 1:
            result.append(list(all_triples[i]))
            
    return result
