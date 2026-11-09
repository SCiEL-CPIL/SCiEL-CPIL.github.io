import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
    
    # Map from triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs
    # For each such configuration, store the 8 triple indices
    k222_configs = []
    # Generate all ways to pick 3 disjoint pairs from 16 points
    # First pick 6 distinct points, then partition into 3 pairs
    for six_points in combinations(range(n), 6):
        # Partition 6 points into 3 pairs: C(6,2)*C(4,2)*C(2,2)/3! = 15*6*1/6 = 15
        a, b, c, d, e, f = six_points
        # All perfect matchings of 6 points
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
        for m in matchings:
            p1, p2, p3 = m
            # 8 transversal triples
            t1 = tuple(sorted((p1[0], p2[0], p3[0])))
            t2 = tuple(sorted((p1[0], p2[0], p3[1])))
            t3 = tuple(sorted((p1[0], p2[1], p3[0])))
            t4 = tuple(sorted((p1[0], p2[1], p3[1])))
            t5 = tuple(sorted((p1[1], p2[0], p3[0])))
            t6 = tuple(sorted((p1[1], p2[0], p3[1])))
            t7 = tuple(sorted((p1[1], p2[1], p3[0])))
            t8 = tuple(sorted((p1[1], p2[1], p3[1])))
            idxs = [triple_to_idx[t] for t in [t1, t2, t3, t4, t5, t6, t7, t8]]
            k222_configs.append(tuple(idxs))
    
    num_k222 = len(k222_configs)
    
    # For each triple index, list which k222 configs it belongs to
    triple_to_k222s = [[] for _ in range(num_triples)]
    for k_idx, cfg in enumerate(k222_configs):
        for t_idx in cfg:
            triple_to_k222s[t_idx].append(k_idx)
    
    # Simulated annealing
    # State: bitmask of selected triples (use Python int as bitset)
    # We'll track violations incrementally
    
    def count_violations(selected_mask):
        """Count number of K(2,2,2) configs with all 8 triples selected."""
        count = 0
        for cfg in k222_configs:
            all_present = True
            for t_idx in cfg:
                if not (selected_mask >> t_idx) & 1:
                    all_present = False
                    break
            if all_present:
                count += 1
        return count
    
    # Initialize: start with a random subset
    # Use a greedy approach to get a good starting point
    selected_mask = 0
    # Try adding triples in random order, skip if it creates a violation
    order = list(range(num_triples))
    random.shuffle(order)
    for t_idx in order:
        # Check if adding this triple would create any violation
        new_mask = selected_mask | (1 << t_idx)
        # Quick check: for each k222 config containing this triple, check if all others are present
        would_violate = False
        for k_idx in triple_to_k222s[t_idx]:
            cfg = k222_configs[k_idx]
            all_others_present = True
            for ot in cfg:
                if ot != t_idx and not (selected_mask >> ot) & 1:
                    all_others_present = False
                    break
            if all_others_present:
                would_violate = True
                break
        if not would_violate:
            selected_mask = new_mask
    
    best_mask = selected_mask
    best_size = bin(selected_mask).count('1')
    
    # Simulated annealing
    temp = 5.0
    cooling = 0.9995
    min_temp = 0.01
    
    while temp > min_temp:
        if time.time() > deadline - 5:
            break
        
        # Pick a random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random unselected triple
            unselected = [i for i in range(num_triples) if not (selected_mask >> i) & 1]
            if not unselected:
                continue
            t_idx = random.choice(unselected)
            new_mask = selected_mask | (1 << t_idx)
            new_size = best_size + 1 if not (selected_mask >> t_idx) & 1 else best_size
            new_size = bin(new_mask).count('1')
            # Check violations
            violations = 0
            for k_idx in triple_to_k222s[t_idx]:
                cfg = k222_configs[k_idx]
                all_present = True
                for ot in cfg:
                    if not (new_mask >> ot) & 1:
                        all_present = False
                        break
                if all_present:
                    violations += 1
            if violations == 0:
                selected_mask = new_mask
                best_size = new_size
                if new_size > best_size:
                    best_mask = selected_mask
            else:
                # Accept with probability exp(-violations/temp)
                if random.random() < (2.718281828 ** (-violations / temp)):
                    selected_mask = new_mask
                    best_size = new_size
        elif move_type < 0.7:
            # Remove a random selected triple
            selected_list = [i for i in range(num_triples) if (selected_mask >> i) & 1]
            if not selected_list:
                continue
            t_idx = random.choice(selected_list)
            new_mask = selected_mask & ~(1 << t_idx)
            new_size = bin(new_mask).count('1')
            # Removing always reduces or keeps violations, so accept
            selected_mask = new_mask
            best_size = new_size
        else:
            # Swap: remove one selected, add one unselected
            selected_list = [i for i in range(num_triples) if (selected_mask >> i) & 1]
            unselected = [i for i in range(num_triples) if not (selected_mask >> i) & 1]
            if not selected_list or not unselected:
                continue
            t_remove = random.choice(selected_list)
            t_add = random.choice(unselected)
            new_mask = (selected_mask & ~(1 << t_remove)) | (1 << t_add)
            new_size = bin(new_mask).count('1')
            # Check violations from the added triple
            violations = 0
            for k_idx in triple_to_k222s[t_add]:
                cfg = k222_configs[k_idx]
                all_present = True
                for ot in cfg:
                    if not (new_mask >> ot) & 1:
                        all_present = False
                        break
                if all_present:
                    violations += 1
            if violations == 0:
                selected_mask = new_mask
                best_size = new_size
                if new_size > best_size:
                    best_mask = selected_mask
            else:
                if random.random() < (2.718281828 ** (-violations / temp)):
                    selected_mask = new_mask
                    best_size = new_size
        
        temp *= cooling
    
    # Final: ensure best_mask has no violations (it should, but verify)
    # Convert back to list of triples
    result = []
    for i in range(num_triples):
        if (best_mask >> i) & 1:
            result.append(list(all_triples[i]))
    
    return result
